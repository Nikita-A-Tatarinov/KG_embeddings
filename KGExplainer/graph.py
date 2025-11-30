# KGExplainer/graph.py
from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import dgl
import torch

# This follows the original dataloader_distill._construct_dgl_graph + _get_subgraph idea

def build_dgl_graph(
    train_triples: torch.LongTensor,
    nentity: int,
    nrelation: int,
    add_reverse: bool = True,
) -> dgl.DGLHeteroGraph:
    """
    Build a DGL heterograph from train triples.

    train_triples: (N,3) LongTensor of (h,r,t)
    node type: 'node'
    edge types: '0', '1', ..., str(r)
    """
    edge_dict = defaultdict(list)
    for h, r, t in train_triples.tolist():
        etype = ("node", str(r), "node")
        edge_dict[etype].append((h, t))
        if add_reverse:
            etype_rev = ("node", f"rev_{r}", "node")
            edge_dict[etype_rev].append((t, h))

    g = dgl.heterograph(edge_dict, num_nodes_dict={"node": nentity})
    return g


def k_hop_enclosing_subgraph(
    g: dgl.DGLHeteroGraph,
    h: int,
    t: int,
    k_hop: int = 2,
) -> dgl.DGLHeteroGraph:
    """
    Approximate GraIL-style enclosing subgraph:
      - collect k-hop neighborhoods around h and t
      - intersect them (plus include h,t themselves)
      - return induced node-subgraph.
    """

    # BFS from h
    h_frontier = {h}
    h_reached = {h}
    for _ in range(k_hop):
        new_nodes = set()
        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            src = src.tolist()
            dst = dst.tolist()
            for u, v in zip(src, dst):
                if u in h_frontier:
                    new_nodes.add(v)
        h_frontier = new_nodes
        h_reached |= new_nodes

    # BFS from t
    t_frontier = {t}
    t_reached = {t}
    for _ in range(k_hop):
        new_nodes = set()
        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            src = src.tolist()
            dst = dst.tolist()
            for u, v in zip(src, dst):
                if u in t_frontier:
                    new_nodes.add(v)
        t_frontier = new_nodes
        t_reached |= new_nodes

    inter_nodes = h_reached & t_reached
    inter_nodes |= {h, t}
    node_ids = list(inter_nodes)

    # node_subgraph expects dict of ntype -> ids
    subg = dgl.node_subgraph(g, {"node": node_ids})
    return subg
