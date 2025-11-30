import os
from collections import defaultdict

import networkx as nx
from collections import defaultdict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .kg_model import KGModel
from .registry import register_model



class ContextPoolingGraph:
    """
    GPU-Accelerated Graph Storage using Torch Sparse CSR.
    Handles neighbor sampling and query masking efficiently on device.
    """


    def __init__(self, triples, n_ent, n_rel, device):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device

        # 1. Prepare triples (Facts + Inverse + Self-loops)
        facts = torch.tensor(triples, dtype=torch.long, device=device)


        inv_facts = torch.stack([facts[:, 2], facts[:, 1] + n_rel, facts[:, 0]], dim=1)


        ents = torch.arange(n_ent, device=device)
        self_loops = torch.stack([ents, torch.full_like(ents, 2 * n_rel), ents], dim=1)


        all_edges = torch.cat([facts, inv_facts, self_loops], dim=0)

        # 2. Build CSR
        sort_idx = torch.argsort(all_edges[:, 0])
        all_edges = all_edges[sort_idx]

        self.src = all_edges[:, 0].contiguous()
        self.rel = all_edges[:, 1].contiguous()
        self.dst = all_edges[:, 2].contiguous()

        counts = torch.bincount(self.src, minlength=n_ent)
        self.indptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(counts, 0)])

    def to(self, device):
        if self.device == device:
            return
        self.indptr = self.indptr.to(device)
        self.src = self.src.to(device)
        self.rel = self.rel.to(device)
        self.dst = self.dst.to(device)
        self.device = device

    def _csr_get_flat_indices(self, starts, lengths):
        starts = starts.contiguous()
        lengths = lengths.contiguous()

        total_edges = lengths.sum().item()
        if total_edges == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        repeat_idx = torch.repeat_interleave(torch.arange(len(lengths), device=self.device), lengths)

        cumsum_len = torch.cumsum(lengths, 0)
        segment_starts = torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), cumsum_len[:-1]])

        global_range = torch.arange(total_edges, device=self.device)
        inner_offsets = global_range - segment_starts[repeat_idx]

        return starts[repeat_idx] + inner_offsets

    @staticmethod
    def generate_cp_stats(triples, n_rel, acc_thresh=0.4, rec_thresh=0.1):
        G = nx.DiGraph()
        relation2neighbors = defaultdict(lambda: defaultdict(int))


        for h, r, t in triples:
            h, r, t = int(h), int(r), int(t)
            r_inv = r + n_rel
            G.add_edge(h, t, relation=r)
            G.add_edge(t, h, relation=r_inv)

        neighbor_num = defaultdict(int)
        for u in G.nodes():
            try:
                edges = G[u]
                rels = [edges[v]["relation"] for v in edges]
                rels = [edges[v]["relation"] for v in edges]
            except Exception:
                continue
            unique_rels = sorted(list(set(rels)))
            for r in unique_rels:
                neighbor_num[r] += 1
                for r2 in unique_rels:
                    relation2neighbors[r][r2] += 1

        accuracy_neighbors = defaultdict(list)
        recall_neighbors = defaultdict(list)
        all_rels = sorted(list(neighbor_num.keys()))

        for r in all_rels:
            for r2 in all_rels:
                cooc = relation2neighbors[r][r2]
                if neighbor_num[r] > 0 and cooc / neighbor_num[r] > acc_thresh:
                    accuracy_neighbors[r].append(r2)
                if neighbor_num[r2] > 0 and cooc / neighbor_num[r2] > rec_thresh:
                    recall_neighbors[r].append(r2)

        dim = 2 * n_rel + 1
        acc_t = torch.zeros((dim, dim), dtype=torch.bool)
        rec_t = torch.zeros((dim, dim), dtype=torch.bool)


        for r, neighbors in accuracy_neighbors.items():
            acc_t[r, neighbors] = True
        for r, neighbors in recall_neighbors.items():
            rec_t[r, neighbors] = True

        return acc_t, rec_t

    def get_neighbors_full(self, nodes, q_sub=None, q_rel=None):
        batch_ids = nodes[:, 0]
        node_ids = nodes[:, 1]


        row_starts = self.indptr[node_ids]
        row_ends = self.indptr[node_ids + 1]
        lengths = row_ends - row_starts


        mask_non_zero = lengths > 0
        if not mask_non_zero.any():
            return torch.empty(0, device=self.device), torch.empty((0, 6), device=self.device), torch.empty(0, device=self.device)

            return torch.empty(0, device=self.device), torch.empty((0, 6), device=self.device), torch.empty(0, device=self.device)

        starts = row_starts[mask_non_zero]
        active_lengths = lengths[mask_non_zero]


        flat_indices = self._csr_get_flat_indices(starts, active_lengths)


        rels = self.rel[flat_indices]
        dsts = self.dst[flat_indices]

        active_indices_in_input = torch.nonzero(mask_non_zero).squeeze(1)
        head_local_idx = active_indices_in_input.repeat_interleave(active_lengths)

        batch_ids_rep = batch_ids[head_local_idx]
        src_ids_rep = node_ids[head_local_idx]

        if q_sub is not None and q_rel is not None:
            q_h = q_sub[batch_ids_rep]
            q_r = q_rel[batch_ids_rep]
            mask = ~((src_ids_rep == q_h) & (rels == q_r))


            batch_ids_rep = batch_ids_rep[mask]
            head_local_idx = head_local_idx[mask]
            src_ids_rep = src_ids_rep[mask]
            rels = rels[mask]
            dsts = dsts[mask]

        sampled_edges = torch.stack([batch_ids_rep, src_ids_rep, rels, dsts], dim=1)


        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_local_idx.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        loop_rel_id = 2 * self.n_rel
        mask_loop = sampled_edges[:, 2] == loop_rel_id
        _, old_idx = head_index[mask_loop].sort()
        old_nodes_new_idx = tail_index[mask_loop][old_idx]


        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_cp_neighbors(self, nodes, query_relations, cp_tensor, head_idx_map, tail_idx_map, complement=False, q_sub=None, q_rel=None):
        batch_ids = nodes[:, 0]
        node_ids = nodes[:, 1]


        row_starts = self.indptr[node_ids]
        row_ends = self.indptr[node_ids + 1]
        lengths = row_ends - row_starts


        mask_non_zero = lengths > 0
        if not mask_non_zero.any():
            return torch.empty(0, device=self.device), torch.empty((0, 6), device=self.device), torch.empty(0, device=self.device)

            return torch.empty(0, device=self.device), torch.empty((0, 6), device=self.device), torch.empty(0, device=self.device)

        starts = row_starts[mask_non_zero]
        active_lengths = lengths[mask_non_zero]

        flat_indices = self._csr_get_flat_indices(starts, active_lengths)
        rels = self.rel[flat_indices]
        dsts = self.dst[flat_indices]

        active_indices_in_input = torch.nonzero(mask_non_zero).squeeze(1)
        head_local_idx = active_indices_in_input.repeat_interleave(active_lengths)

        batch_ids_rep = batch_ids[head_local_idx]

        relations_next = rels
        is_self_loop = relations_next == self.n_rel * 2
        is_self_loop = relations_next == self.n_rel * 2
        relations_next_inv = (relations_next + self.n_rel) % (2 * self.n_rel)
        relations_next_inv[is_self_loop] = self.n_rel * 2


        q_rels = query_relations[batch_ids_rep]
        cp_mask = cp_tensor[q_rels, relations_next_inv]
        if complement:
            cp_mask = ~cp_mask


        final_mask = cp_mask | is_self_loop


        if q_sub is not None and q_rel is not None:
            q_h = q_sub[batch_ids_rep]
            q_r = q_rel[batch_ids_rep]
            src_ids_rep = node_ids[head_local_idx]
            q_mask = ~((src_ids_rep == q_h) & (rels == q_r))
            final_mask = final_mask & q_mask


        valid_indices = torch.nonzero(final_mask).squeeze()
        if valid_indices.numel() == 0:
            return torch.empty(0, device=self.device), torch.empty((0, 6), device=self.device), torch.empty(0, device=self.device)

            return torch.empty(0, device=self.device), torch.empty((0, 6), device=self.device), torch.empty(0, device=self.device)

        batch_ids_rep = batch_ids_rep[valid_indices]
        src_ids_rep = node_ids[head_local_idx][valid_indices]  # Reconstruct correct head
        rels = rels[valid_indices]
        dsts = dsts[valid_indices]


        sampled_edges = torch.stack([batch_ids_rep, src_ids_rep, rels, dsts], dim=1)


        head_index = head_idx_map[sampled_edges[:, 0], sampled_edges[:, 1]]
        tail_index = tail_idx_map[sampled_edges[:, 0], sampled_edges[:, 3]]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        tail_nodes = None

        return tail_nodes, sampled_edges, None


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super().__init__()
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super().__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, crop_dim=None):
        if crop_dim is not None:
            rel_emb_w = self.rela_embed.weight[:, :crop_dim]
            sub, rel, obj = edges[:, 4], edges[:, 2], edges[:, 5]


            hs = hidden[sub]
            hr = F.embedding(rel, rel_emb_w)


            r_idx = edges[:, 0]
            q_rel_emb = F.embedding(q_rel, rel_emb_w)
            h_qr = q_rel_emb[r_idx]

            ws_w = self.Ws_attn.weight[:, :crop_dim]
            wr_w = self.Wr_attn.weight[:, :crop_dim]
            wqr_w = self.Wqr_attn.weight[:, :crop_dim]
            wqr_b = self.Wqr_attn.bias

            term1 = F.linear(hs, ws_w)
            term2 = F.linear(hr, wr_w)
            term3 = F.linear(h_qr, wqr_w, wqr_b)

            message = hs + hr
            alpha_in = F.relu(term1 + term2 + term3)
            alpha = torch.sigmoid(self.w_alpha(alpha_in))
            message = alpha * message

            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce="sum")
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce="sum")
            wh_w = self.W_h.weight[:crop_dim, :crop_dim]
            hidden_new = self.act(F.linear(message_agg, wh_w))
            return hidden_new

        sub, rel, obj = edges[:, 4], edges[:, 2], edges[:, 5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]


        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce="sum")

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce="sum")
        hidden_new = self.act(self.W_h(message_agg))
        return hidden_new


@register_model("ContextPooling", "cp")
class ContextPooling(KGModel):
    ENTITY_FACTOR = 1
    RELATION_FACTOR = 1

    def __init__(
        self,
        nentity,
        nrelation,
        base_dim,
        gamma,
        n_layer=2,
        attn_dim=5,
        dropout=0.1,
        act="relu",
        train_triples=None,
        data_path=None,
        accuracy_threshold=0.4,
        recall_threshold=0.1,
        **kwargs,
    ):
        super().__init__(nentity, nrelation, base_dim, gamma, **kwargs)


        self.hidden_dim = base_dim
        self.attn_dim = attn_dim
        self.n_layer = int(n_layer)
        self.dropout = nn.Dropout(dropout)

        activations = {"relu": nn.ReLU(), "tanh": torch.tanh, "idd": lambda x: x}
        self.act = activations.get(act, nn.ReLU())

        if train_triples is not None:
            self._init_graph(train_triples, accuracy_threshold, recall_threshold)
        elif data_path is not None:
            self._load_and_init_graph(data_path)

        self.gnn_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.acc_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.rec_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.acc_c_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.rec_c_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )

            self._load_and_init_graph(data_path)

        self.gnn_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.acc_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.rec_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.acc_c_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )
        self.rec_c_layers = nn.ModuleList(
            [GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, nrelation, self.act) for _ in range(self.n_layer)]
        )

        self.gate = nn.GRU(self.hidden_dim * 5, self.hidden_dim)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

    @property
    def device(self):
        return self.entity_embedding.device

    def _load_and_init_graph(self, data_path):
        path = os.path.join(data_path, "train.txt")
        path = os.path.join(data_path, "train.txt")
        triples = []
        with open(path) as f:
        with open(path) as f:
            for line in f:
                h, r, t = line.strip().split()
                triples.append([int(h), int(r), int(t)])
        self._init_graph(triples)

    def _init_graph(self, triples, acc_thresh, rec_thresh):
        if isinstance(triples, torch.Tensor):
            triples = triples.cpu().numpy()

        acc_t, rec_t = ContextPoolingGraph.generate_cp_stats(triples, self.nrelation, acc_thresh, rec_thresh)
        self.register_buffer("accuracy_tensor", acc_t)
        self.register_buffer("recall_tensor", rec_t)

        self.graph = ContextPoolingGraph(triples, self.nentity, self.nrelation, self.device)

    def _sliced_gru_step(self, x, h_prev, crop_dim):
        hidden_full = self.hidden_dim


        idx_r = slice(0, crop_dim)
        idx_z = slice(hidden_full, hidden_full + crop_dim)
        idx_n = slice(2 * hidden_full, 2 * hidden_full + crop_dim)

        idx_n = slice(2 * hidden_full, 2 * hidden_full + crop_dim)

        def get_sliced_w(w_full, in_dim_slice):
            return torch.cat([w_full[idx_r, :in_dim_slice], w_full[idx_z, :in_dim_slice], w_full[idx_n, :in_dim_slice]], dim=0)

        def get_sliced_b(b_full):
            return torch.cat([b_full[idx_r], b_full[idx_z], b_full[idx_n]], dim=0)

        w_ih = get_sliced_w(self.gate.weight_ih_l0, 5 * crop_dim)
        b_ih = get_sliced_b(self.gate.bias_ih_l0)
        w_hh = get_sliced_w(self.gate.weight_hh_l0, crop_dim)
        b_hh = get_sliced_b(self.gate.bias_hh_l0)


        gi = F.linear(x, w_ih, b_ih)
        gh = F.linear(h_prev, w_hh, b_hh)


        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)


        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)


        hy = newgate + inputgate * (h_prev - newgate)
        return hy

    def forward_gnn(self, subs, rels, crop_dim=None, mode="train"):
    def forward_gnn(self, subs, rels, crop_dim=None, mode="train"):
        if self.graph is None:
            raise RuntimeError("Graph not initialized.")

        if self.graph.device != self.device:
            self.graph.to(self.device)

        q_sub = subs.to(self.device)
        q_rel = rels.to(self.device)
        batch_size = len(q_sub)

        mask_q_sub = q_sub if mode != "test" else None
        mask_q_rel = q_rel if mode != "test" else None

        mask_q_sub = q_sub if mode != "test" else None
        mask_q_rel = q_rel if mode != "test" else None

        dim = crop_dim if crop_dim is not None else self.hidden_dim


        nodes = torch.cat([torch.arange(batch_size, device=self.device).unsqueeze(1), q_sub.unsqueeze(1)], 1)
        # Use ZEROS for initialization (Official CP implementation)
        h0 = torch.zeros((batch_size, dim), device=self.device)
        hidden = torch.zeros(batch_size, dim, device=self.device)


        inv_rels = (q_rel + self.nrelation) % (2 * self.nrelation)
        q_rels_acc = inv_rels
        q_rels_rec = inv_rels
        q_rels_acc_c = inv_rels
        q_rels_rec_c = inv_rels

        for i in range(self.n_layer):
            hidden_list = []

            nodes_full, edges_full, old_nodes_new_idx = self.graph.get_neighbors_full(nodes, q_sub=mask_q_sub, q_rel=mask_q_rel)


            nodes_full, edges_full, old_nodes_new_idx = self.graph.get_neighbors_full(nodes, q_sub=mask_q_sub, q_rel=mask_q_rel)

            head_idx_map = torch.zeros((batch_size, self.nentity), dtype=torch.long, device=self.device)
            tail_idx_map = torch.zeros((batch_size, self.nentity), dtype=torch.long, device=self.device)


            head_idx_map[edges_full[:, 0], edges_full[:, 1]] = edges_full[:, 4]
            tail_idx_map[edges_full[:, 0], edges_full[:, 3]] = edges_full[:, 5]


            h_full = self.gnn_layers[i](q_sub, q_rel, hidden, edges_full, nodes_full.size(0), crop_dim=crop_dim)
            hidden_list.append(h_full)

            def run_stream(layer, q_rels_prev, tensor, complement, nodes_curr, head_map, tail_map, hidden_curr, full_emb, nodes_full_curr):

            def run_stream(layer, q_rels_prev, tensor, complement, nodes_curr, head_map, tail_map, hidden_curr, full_emb, nodes_full_curr):
                _, edges, _ = self.graph.get_cp_neighbors(
                    nodes_curr, q_rels_prev, tensor, head_map, tail_map, complement=complement, q_sub=mask_q_sub, q_rel=mask_q_rel
                    nodes_curr, q_rels_prev, tensor, head_map, tail_map, complement=complement, q_sub=mask_q_sub, q_rel=mask_q_rel
                )
                if edges.size(0) == 0:
                    return torch.zeros_like(full_emb)
                return layer(q_sub, q_rel, hidden_curr, edges, nodes_full_curr.size(0), crop_dim=crop_dim)
                    return torch.zeros_like(full_emb)
                return layer(q_sub, q_rel, hidden_curr, edges, nodes_full_curr.size(0), crop_dim=crop_dim)

            hidden_list.append(
                run_stream(
                    self.acc_layers[i],
                    q_rels_acc,
                    self.accuracy_tensor,
                    False,
                    nodes,
                    head_idx_map,
                    tail_idx_map,
                    hidden,
                    h_full,
                    nodes_full,
                )
            )
            hidden_list.append(
                run_stream(
                    self.rec_layers[i], q_rels_rec, self.recall_tensor, False, nodes, head_idx_map, tail_idx_map, hidden, h_full, nodes_full
                )
            )
            hidden_list.append(
                run_stream(
                    self.acc_c_layers[i],
                    q_rels_acc_c,
                    self.accuracy_tensor,
                    True,
                    nodes,
                    head_idx_map,
                    tail_idx_map,
                    hidden,
                    h_full,
                    nodes_full,
                )
            )
            hidden_list.append(
                run_stream(
                    self.rec_c_layers[i],
                    q_rels_rec_c,
                    self.recall_tensor,
                    True,
                    nodes,
                    head_idx_map,
                    tail_idx_map,
                    hidden,
                    h_full,
                    nodes_full,
                )
            )

            hidden_combined = torch.cat(hidden_list, dim=1)


            h0_expanded = torch.zeros(nodes_full.size(0), dim, device=self.device)
            h0_expanded.index_copy_(0, old_nodes_new_idx, h0)


            hidden_combined = self.dropout(hidden_combined)


            if crop_dim is not None:
                hidden = self._sliced_gru_step(hidden_combined, h0_expanded, crop_dim)
            else:
                out, _ = self.gate(hidden_combined.unsqueeze(0), h0_expanded.unsqueeze(0))
                hidden = out.squeeze(0)


            h0 = hidden
            nodes = nodes_full

        if crop_dim is not None:
            w_final = self.W_final.weight[:, :dim]
            scores = F.linear(hidden, w_final).squeeze(-1)
        else:
            scores = self.W_final(hidden).squeeze(-1)


        scores_all = torch.zeros((batch_size, self.nentity), device=self.device)
        scores_all[nodes_full[:, 0], nodes_full[:, 1]] = scores
        return scores_all

    def forward(self, sample, mode="single", crop_dim=None):
        gnn_mode = "train" if self.training else "test"
    def forward(self, sample, mode="single", crop_dim=None):
        gnn_mode = "train" if self.training else "test"

        if mode == "single":
        if mode == "single":
            h_idx, r_idx, t_idx = sample[:, 0], sample[:, 1], sample[:, 2]
            all_scores = self.forward_gnn(h_idx, r_idx, crop_dim=crop_dim, mode=gnn_mode)
            return all_scores[torch.arange(len(sample), device=self.device), t_idx].unsqueeze(1)

        elif mode == "head-batch":

        elif mode == "head-batch":
            tail_part, head_part = sample
            r_idx = tail_part[:, 1]
            t_idx = tail_part[:, 2]
            r_inv_idx = (r_idx + self.nrelation) % (2 * self.nrelation)
            all_scores = self.forward_gnn(t_idx, r_inv_idx, crop_dim=crop_dim, mode=gnn_mode)
            return torch.gather(all_scores, 1, head_part.to(self.device))

        elif mode == "tail-batch":
        elif mode == "tail-batch":
            head_part, tail_part = sample
            h_idx = head_part[:, 0]
            r_idx = head_part[:, 1]
            all_scores = self.forward_gnn(h_idx, r_idx, crop_dim=crop_dim, mode=gnn_mode)
            return torch.gather(all_scores, 1, tail_part.to(self.device))

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        device = self.device
        head = head.to(device)
        relation = relation.to(device)
        tail = tail.to(device)

        gnn_mode = "train" if self.training else "test"

        if mode == "single":
            all_scores = self.forward_gnn(head, relation, crop_dim=crop_dim, mode=gnn_mode)
            batch_idx = torch.arange(head.size(0), device=device)
            return all_scores[batch_idx, tail].unsqueeze(1)

        elif mode == "head-batch":
            r_inv = (relation + self.nrelation) % (2 * self.nrelation)
            all_scores = self.forward_gnn(tail, r_inv, crop_dim=crop_dim, mode=gnn_mode)
            return torch.gather(all_scores, 1, head)

        elif mode == "tail-batch":
            all_scores = self.forward_gnn(head, relation, crop_dim=crop_dim, mode=gnn_mode)
            return torch.gather(all_scores, 1, tail)

        else:
            raise ValueError(f"Unknown mode: {mode}")
