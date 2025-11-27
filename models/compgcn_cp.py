# models/compgcn_cp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .kg_model import KGModel
from .registry import register_model
from dataset.utils import load_kg_hf

# Import existing models for Mixin
from .transe import TransE
from .distmult import DistMult
from .rotate import RotatE
from .complex_ import ComplEx
from .pairre import PairRE

class CompGCNBase(KGModel):
    """
    Base class for CompGCN with Context Pooling.
    Handles:
      1. Loading CNF and Graph Structure.
      2. GNN Propagation (Context Pooling).
      3. Forward pass overrides to inject GNN embeddings before scoring.
    Does NOT implement score() - that comes from the Mixin (e.g. TransE).
    """
    def __init__(self, nentity, nrelation, base_dim, gamma, **kwargs):
        # Initialize parent (KGModel)
        super().__init__(nentity, nrelation, base_dim, gamma, **kwargs)
        
        # --- 1. Load Data Resources ---
        self.args = kwargs.get('args', None)
        # Fallback to defaults if args not present (e.g. in simple tests)
        dataset_name = kwargs.get('dataset', 'FB15k-237')
        data_path = kwargs.get('data_path', './data')
        
        self._load_cnf(data_path, dataset_name)
        self._load_graph_structure(dataset_name)

        # --- 2. GNN Layers ---
        self.hidden_dim = int(base_dim)
        
        # We stick to simple aggregation: Base || Context -> Fusion -> Output
        self.W_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.Tanh()
        
        # Optional: Transformation for relations if needed by CompGCN logic
        # self.W_rel = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _load_cnf(self, data_path, dataset_name):
        """Loads the pre-computed CNF dictionary."""
        cnf_path = os.path.join(data_path, dataset_name, "cnf.pt")
        if os.path.exists(cnf_path):
            # Load to CPU first. We move specific tensors to GPU on demand during forward
            # to save VRAM, or move all now if graph is small.
            self.cnf = torch.load(cnf_path, map_location='cpu')
            print(f"[CompGCN_CP] Loaded CNF from {cnf_path}")
        else:
            print(f"[CompGCN_CP] WARNING: CNF not found at {cnf_path}. Context Pooling will be disabled.")
            self.cnf = {}

    def _load_graph_structure(self, dataset_name):
        """Loads training triples to build adjacency for GNN."""
        # Map common names to HF repos if needed, or assume local if prefer_ids logic used
        # Here we reuse the load_kg_hf utility for consistency
        hf_repo = f"KGraph/{dataset_name}" if dataset_name in ["FB15k-237", "WN18RR", "NELL-995"] else dataset_name
        
        try:
            # We only need train_ids to build the graph
            train_ids, _, _, _, _ = load_kg_hf(hf_repo)
            
            heads = train_ids[:, 0]
            rels = train_ids[:, 1]
            tails = train_ids[:, 2]
            
            # Add Inverse Edges for message passing
            # Inverse R ID = R + nrelation
            inv_heads = tails
            inv_tails = heads
            inv_rels = rels + self.nrelation
            
            all_heads = torch.cat([heads, inv_heads])
            all_tails = torch.cat([tails, inv_tails])
            all_rels = torch.cat([rels, inv_rels])
            
            # Register as buffer so it moves to device with model
            self.register_buffer('edge_index', torch.stack([all_heads, all_tails])) # (2, E)
            self.register_buffer('edge_type', all_rels) # (E,)
            
            print(f"[CompGCN_CP] Graph initialized with {self.edge_index.shape[1]} edges (including inverse).")
            
        except Exception as e:
            print(f"[CompGCN_CP] Error loading graph: {e}. GNN will be no-op.")
            self.register_buffer('edge_index', torch.zeros((2, 0), dtype=torch.long))
            self.register_buffer('edge_type', torch.zeros((0,), dtype=torch.long))

    def forward(self, sample, mode='single', crop_dim=None):
        """
        Override forward to perform GNN updates on Anchors before scoring.
        """
        # 1. Get Static Embeddings (Standard Lookup)
        # This gives us the base representations for H, R, T
        # head_emb, tail_emb: (B, 1, D) or (B, K, D)
        head_emb, relation_emb, tail_emb, rel_ids = self._index_full(sample, mode)
        
        # 2. Refine Anchors using Context Pooling
        # We need the indices to query the graph.
        
        if mode == 'single':
            # sample is (B, 3). H and T are both anchors.
            h_idx, r_idx, t_idx = sample[:, 0], sample[:, 1], sample[:, 2]
            
            # Update Head using r context
            h_gnn = self._gnn_context_step(h_idx, r_idx, head_emb.squeeze(1), inverse=False)
            h_emb = h_gnn.unsqueeze(1)
            
            # Update Tail using r_inv context
            t_gnn = self._gnn_context_step(t_idx, r_idx, tail_emb.squeeze(1), inverse=True)
            tail_emb = t_gnn.unsqueeze(1)
            
        elif mode == 'head-batch':
            # sample = (tail_part, head_part)
            # We predict Head given (r, t). Anchor is Tail.
            tail_part, _ = sample
            r_idx = tail_part[:, 1]
            t_idx = tail_part[:, 2]
            
            # Update Tail using r_inv context
            # We use the static tail_emb provided by _index_full as the base
            t_base = tail_emb.squeeze(1) # (B, D)
            t_gnn = self._gnn_context_step(t_idx, r_idx, t_base, inverse=True)
            tail_emb = t_gnn.unsqueeze(1)
            
            # Candidates (Heads) remain static to save computation
            
        elif mode == 'tail-batch':
            # sample = (head_part, tail_part)
            # We predict Tail given (h, r). Anchor is Head.
            head_part, _ = sample
            h_idx = head_part[:, 0]
            r_idx = head_part[:, 1]
            
            # Update Head using r context
            h_base = head_emb.squeeze(1)
            h_gnn = self._gnn_context_step(h_idx, r_idx, h_base, inverse=False)
            head_emb = h_gnn.unsqueeze(1)

        # 3. Score using the Mixin's implementation of score()
        # The Mixin (e.g. TransE) expects embeddings, which we now provide.
        return self.score(head_emb, relation_emb, tail_emb, mode, rel_ids=rel_ids, crop_dim=crop_dim)

    def _gnn_context_step(self, anchors, query_rels, base_embs, inverse=False):
        """
        Performs Context Pooling aggregation for a batch of anchors.
        
        Args:
            anchors: (B,) indices of anchor entities
            query_rels: (B,) indices of the query relation (raw ID)
            base_embs: (B, D) static embeddings of anchors
            inverse: Bool, true if we are looking for context of inverse relation
        """
        device = base_embs.device
        batch_size = anchors.size(0)
        
        # Accumulator for context aggregation
        agg_context = torch.zeros_like(base_embs)
        
        # We process unique query relations in the batch to vectorize neighbor lookup
        # (Since CNF is different for each query relation)
        unique_rels, inverse_indices = torch.unique(query_rels, return_inverse=True)
        
        for i, r_type_idx in enumerate(unique_rels):
            r_id = r_type_idx.item()
            
            # 1. Identify which samples in batch match this relation
            batch_mask = (inverse_indices == i)
            sub_anchors = anchors[batch_mask]
            
            # 2. Get Context Relations from CNF
            # If inverse=True (e.g. Tail anchor), we need context for r_inv
            # r_inv ID = r_id + nrelation
            query_r_cnf_id = r_id + self.nrelation if inverse else r_id
            
            if query_r_cnf_id in self.cnf:
                allowed_rels = self.cnf[query_r_cnf_id].to(device)
            else:
                allowed_rels = torch.empty(0, device=device, dtype=torch.long)
                
            if len(allowed_rels) == 0:
                continue

            # 3. Build Query-Specific Subgraph (Filter Edges)
            # We need edges: Source == sub_anchors AND Relation in allowed_rels
            # Note: GNN message flows Neighbors -> Anchor. 
            # In edge_index, if we want neighbors of 'h', we look for edges (h, t).
            # Then we aggregate t's features into h.
            
            # Mask A: Source is in our anchor set
            # optimized: torch.isin is available in newer torch, else use bucket logic
            src_mask = torch.isin(self.edge_index[0], sub_anchors)
            
            if not src_mask.any():
                continue
                
            # Mask B: Relation is logically relevant
            # We apply this only to the edges found in step A to save compute
            candidate_edges = self.edge_index[:, src_mask]
            candidate_types = self.edge_type[src_mask]
            
            rel_mask = torch.isin(candidate_types, allowed_rels)
            
            if not rel_mask.any():
                continue
                
            final_edges = candidate_edges[:, rel_mask]
            final_types = candidate_types[rel_mask]
            
            # 4. Aggregate Messages (Neighbor Embeddings * Relation Embeddings)
            # Destination nodes (neighbors) are at index 1
            neighbor_emb = F.embedding(final_edges[1], self.entity_embedding)
            edge_rel_emb = F.embedding(final_types, self.relation_embedding)
            
            # Composition function phi(e, r). Simple element-wise for generic support.
            messages = neighbor_emb * edge_rel_emb 
            
            # 5. Scatter Sum back to Batch
            # We need to sum messages corresponding to each anchor.
            # final_edges[0] contains the global entity IDs of anchors.
            # We can use index_add_ to a global buffer, then extract.
            # (Using a dense global buffer is memory-heavy but simplest for correctness)
            
            # Optimization: scatter to local batch index?
            # We need to map global ID -> index in 'sub_anchors'
            # Since 'sub_anchors' might contain duplicates (if batch has same h repeated),
            # treating them as distinct batch items is tricky with scatter.
            
            # Safer approach: Global Scatter -> Gather
            global_agg = torch.zeros(self.nentity, self.hidden_dim, device=device)
            global_agg.index_add_(0, final_edges[0], messages)
            
            # Extract relevant rows
            batch_updates = F.embedding(sub_anchors, global_agg)
            
            # Add to main accumulator
            agg_context[batch_mask] = batch_updates
            
        # 6. Fusion (Base + Context)
        combined = torch.cat([base_embs, agg_context], dim=-1)
        update = self.dropout(self.act(self.W_fusion(combined)))
        
        return base_embs + update

# --- Concrete Model Implementations via Mixins ---

@register_model("CompGCN_TransE")
class CompGCN_TransE(CompGCNBase, TransE):
    """CompGCN with Context Pooling + TransE Scoring"""
    pass

@register_model("CompGCN_DistMult")
class CompGCN_DistMult(CompGCNBase, DistMult):
    """CompGCN with Context Pooling + DistMult Scoring"""
    pass

@register_model("CompGCN_RotatE")
class CompGCN_RotatE(CompGCNBase, RotatE):
    """CompGCN with Context Pooling + RotatE Scoring"""
    pass

@register_model("CompGCN_ComplEx")
class CompGCN_ComplEx(CompGCNBase, ComplEx):
    """CompGCN with Context Pooling + ComplEx Scoring"""
    pass

@register_model("CompGCN_PairRE")
class CompGCN_PairRE(CompGCNBase, PairRE):
    """CompGCN with Context Pooling + PairRE Scoring"""
    pass