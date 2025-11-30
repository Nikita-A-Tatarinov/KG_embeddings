# This is basically the original HeteroGAT.
from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each


class HeteroGAT(nn.Module):
    """
    Distilled evaluator GNN:
      - Takes a (batched) enclosing subgraph as DGL heterograph
      - Outputs:
          embed: pooled graph embedding (for optional embedding distillation)
          score: scalar logit per graph (triple score)
    """

    def __init__(
        self,
        etypes,
        num_nodes: int,
        in_size: int = 32,
        hid_size: int = 64,
        out_size: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.embed = nn.Embedding(num_nodes, in_size)

        # Three GAT layers, relation-typed via HeteroGraphConv
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(in_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )

        self.dropout = nn.Dropout(0.5)
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(hid_size, out_size),
                nn.Linear(out_size, 1),
                nn.Identity(),  # keep plain score; you can add LogSigmoid if you like
            ]
        )

    def forward(self, g: dgl.DGLHeteroGraph):
        """
        g is a (possibly batched) heterograph with node type 'node'.
        We use node IDs (dgl.NID) to index into self.embed.
        """
        h = {"node": self.embed(g.ndata[dgl.NID])}  # (N_nodes, in_size)

        for l, layer in enumerate(self.layers):
            h = layer(g, h)  # each value: (N_nodes, n_heads, head_dim)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        g.ndata["feat"] = h["node"]
        embed = dgl.sum_nodes(g, "feat")  # (N_graphs, hid_size)

        score = embed
        for layer in self.output_layers:
            score = layer(score)  # (N_graphs, 1)

        return embed, score.squeeze(-1)  # (B, D), (B,)
