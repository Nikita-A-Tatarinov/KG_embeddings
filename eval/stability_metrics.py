"""
Stability metrics for Knowledge Graph Embedding models.

Based on: "Stability and Generalization Capability of Subgraph Reasoning Models 
for Inductive Knowledge Graph Completion" (ICML 2025)

Implements:
1. Empirical Lipschitz Constant (η_f) - measures model sensitivity
2. Stability (C_f = 1/η_f) - higher is better for generalization
3. RTMD (Relational Tree Mover's Distance) - distance metric for subgraphs
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Subgraph:
    """Represents a subgraph extracted around a query triplet."""
    
    def __init__(
        self,
        entities: List[int],
        edges: List[Tuple[int, int, int]],  # (head, relation, tail)
        query_triplet: Tuple[int, int, int],  # (h, r, t)
        initial_embeddings: Dict[int, torch.Tensor],
        nentity: int,
        nrelation: int
    ):
        self.entities = list(entities)
        self.edges = edges
        self.head, self.query_relation, self.tail = query_triplet
        self.initial_embeddings = initial_embeddings
        self.nentity = nentity
        self.nrelation = nrelation
        
        # Build incoming neighbor structure for RTMD computation
        # neighbors[entity] = [(relation, source_entity), ...]
        self.neighbors = defaultdict(list)
        for h, r, t in edges:
            self.neighbors[t].append((r, h))


class SubgraphExtractor:
    """
    Extracts k-hop enclosing subgraphs for triplets.
    
    Based on paper Appendix E: "we extract 2-hop enclosing subgraphs for all 
    positive and negative triplets"
    """
    
    def __init__(
        self, 
        kg_index,  # KGIndex object
        nentity: int,
        nrelation: int,
        k_hops: int = 2, 
        max_neighbors: int = 50
    ):
        """
        Args:
            kg_index: KGIndex object containing all triples
            nentity: Total number of entities
            nrelation: Total number of relations
            k_hops: Number of hops for subgraph extraction
            max_neighbors: Max neighbors per hop (for efficiency)
        """
        self.kg_index = kg_index
        self.nentity = nentity
        self.nrelation = nrelation
        self.k_hops = k_hops
        self.max_neighbors = max_neighbors
        
        # Build adjacency structure from kg_index
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency lists for efficient neighbor queries."""
        # Incoming edges: entity -> [(relation, source_entity), ...]
        self.incoming_edges = defaultdict(list)
        # Outgoing edges: entity -> [(relation, target_entity), ...]
        self.outgoing_edges = defaultdict(list)
        
        # Extract from kg_index.hr2t and kg_index.tr2h
        # hr2t: (h, r) -> {t, ...}
        for (h, r), tails in self.kg_index.hr2t.items():
            for t in tails:
                self.outgoing_edges[h].append((r, t))
                self.incoming_edges[t].append((r, h))
    
    def extract(self, head: int, relation: int, tail: int) -> Subgraph:
        """
        Extract k-hop enclosing subgraph for triplet (h, r, t).
        
        Returns subgraph as intersection of k-hop neighbors from head and tail.
        
        Args:
            head: Head entity ID
            relation: Relation ID
            tail: Tail entity ID
            
        Returns:
            Subgraph object containing structure and initial embeddings
        """
        # 1. Get k-hop neighbors of head (using outgoing edges)
        head_neighbors = self._get_k_hop_neighbors(head, self.k_hops, use_outgoing=True)
        
        # 2. Get k-hop neighbors of tail (using incoming edges)
        tail_neighbors = self._get_k_hop_neighbors(tail, self.k_hops, use_outgoing=False)
        
        # 3. Compute intersection (enclosing subgraph)
        subgraph_entities = head_neighbors.intersection(tail_neighbors)
        subgraph_entities.add(head)
        subgraph_entities.add(tail)
        
        # 4. Extract edges within subgraph
        subgraph_edges = self._get_edges_within(subgraph_entities)
        
        # 5. Create initial embeddings (double radius labeling)
        initial_embeddings = self._create_initial_embeddings(
            subgraph_entities, head, tail
        )
        
        return Subgraph(
            entities=list(subgraph_entities),
            edges=subgraph_edges,
            query_triplet=(head, relation, tail),
            initial_embeddings=initial_embeddings,
            nentity=self.nentity,
            nrelation=self.nrelation
        )
    
    def _get_k_hop_neighbors(self, entity: int, k: int, use_outgoing: bool = True) -> set:
        """
        Get k-hop neighbors of an entity using BFS.
        
        Args:
            entity: Starting entity
            k: Number of hops
            use_outgoing: If True, use outgoing edges; else use incoming edges
        """
        neighbors = set()
        current_level = {entity}
        
        edges_dict = self.outgoing_edges if use_outgoing else self.incoming_edges
        
        for _ in range(k):
            next_level = set()
            for e in current_level:
                # Get neighbors (limit to max_neighbors for efficiency)
                neighbor_list = edges_dict.get(e, [])[:self.max_neighbors]
                for r, neighbor in neighbor_list:
                    next_level.add(neighbor)
            neighbors.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return neighbors
    
    def _get_edges_within(self, entities: set) -> List[Tuple[int, int, int]]:
        """Get all edges where both head and tail are in the subgraph entities."""
        edges = []
        for h, r_set in self.kg_index.hr2t.items():
            head_ent, rel = h
            if head_ent in entities:
                for tail_ent in r_set:
                    if tail_ent in entities:
                        edges.append((head_ent, rel, tail_ent))
        return edges
    
    def _create_initial_embeddings(
        self, entities: set, head: int, tail: int
    ) -> Dict[int, torch.Tensor]:
        """
        Create initial embeddings using double radius vertex labeling.
        
        Based on GraIL: encodes shortest path distance from head and tail
        as one-hot vectors.
        
        UPDATED: Add scaling factor to make embeddings more distinguishable.
        """
        # Compute shortest distances
        dist_from_head = self._bfs_distances(head, entities)
        dist_from_tail = self._bfs_distances(tail, entities)
        
        embeddings = {}
        max_dist = 10  # Maximum distance we encode
        
        for e in entities:
            # One-hot encoding for distance from head (capped at max_dist)
            d_h = min(dist_from_head.get(e, max_dist), max_dist - 1)
            # One-hot encoding for distance from tail
            d_t = min(dist_from_tail.get(e, max_dist), max_dist - 1)
            
            # Create one-hot vectors
            onehot_h = torch.zeros(max_dist)
            onehot_t = torch.zeros(max_dist)
            onehot_h[d_h] = 1.0
            onehot_t[d_t] = 1.0
            
            # Concatenate (no scaling - follows paper exactly)
            embedding = torch.cat([onehot_h, onehot_t])
            
            embeddings[e] = embedding
        
        # Add virtual entity embedding (distinct from all others)
        # Per Definition B.1: blank trees have "a unique label distinct from all other entities"
        VIRTUAL_ENTITY = -1
        embeddings[VIRTUAL_ENTITY] = torch.zeros(20)
        
        return embeddings
    
    def _bfs_distances(self, start: int, entities: set) -> Dict[int, int]:
        """Compute shortest path distances from start to all entities in set."""
        distances = {start: 0}
        queue = [start]
        visited = {start}
        
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            # Explore both incoming and outgoing edges
            for r, neighbor in self.incoming_edges[current] + self.outgoing_edges[current]:
                if neighbor in entities and neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        return distances


def compute_rtmd(
    subgraph1: Subgraph,
    subgraph2: Subgraph,
    num_layers: int = 3
) -> float:
    """
    Compute Relational Tree Mover's Distance between two subgraphs.
    """
    # Virtual root relation for computation trees
    R_ROOT = -1
    
    # Step 1: Build relational computation trees for all entities
    trees_s1 = {
        e: _build_relational_tree(subgraph1, e, num_layers)
        for e in subgraph1.entities
    }
    trees_s2 = {
        e: _build_relational_tree(subgraph2, e, num_layers)
        for e in subgraph2.entities
    }
    
    # Step 2: Compute RTD for head entities
    head_rtd = _compute_rtd(
        (R_ROOT, trees_s1[subgraph1.head]),
        (R_ROOT, trees_s2[subgraph2.head]),
        subgraph1, subgraph2
    )
    
    # Step 3: Compute RTD for tail entities
    tail_rtd = _compute_rtd(
        (R_ROOT, trees_s1[subgraph1.tail]),
        (R_ROOT, trees_s2[subgraph2.tail]),
        subgraph1, subgraph2
    )
    
    # Step 4: Compute OT distance for all entities
    # Per Appendix E: "we compute the exact solution using the POT library"
    # for the outer RTMD OT term (not the inner RTD OT terms)
    all_trees_s1 = [(R_ROOT, trees_s1[e]) for e in subgraph1.entities]
    all_trees_s2 = [(R_ROOT, trees_s2[e]) for e in subgraph2.entities]
    
    # Apply blank tree augmentation (Definition B.1)
    all_trees_s1, all_trees_s2 = _blank_tree_augmentation(all_trees_s1, all_trees_s2)
    
    # Compute optimal transport using EXACT solver for outer RTMD term
    all_entities_ot = _optimal_transport_rtd(
        all_trees_s1, all_trees_s2, subgraph1, subgraph2, use_exact_ot=True
    )
    
    # Step 5: Sum all components per Definition 4.3
    # RTMD(S1, S2) = RTD(head_trees) + RTD(tail_trees) + OT_RTD(all_trees)
    rtmd = head_rtd + tail_rtd + all_entities_ot
    
    return rtmd


def _build_relational_tree(subgraph: Subgraph, entity: int, depth: int):
    """
    Build depth-L relational computation tree for an entity.
    
    Based on Definition 4.1 (Page 4):
    T_S^(l)(v) = (v, SUB(T_S^(l)(v)))
    where SUB contains (relation, subtree) pairs for all neighbors
    """
    if depth == 0:
        return (entity, [])
    
    # Get incoming neighbors
    neighbors = subgraph.neighbors.get(entity, [])
    
    # Recursively build subtrees
    subtrees = []
    for relation, neighbor_entity in neighbors:
        neighbor_tree = _build_relational_tree(subgraph, neighbor_entity, depth - 1)
        subtrees.append((relation, neighbor_tree))
    
    return (entity, subtrees)


def _compute_rtd(
    tree_pair1: Tuple,
    tree_pair2: Tuple,
    subgraph1: Subgraph,
    subgraph2: Subgraph
) -> float:
    """
    Compute Relational Tree Distance between two (relation, tree) pairs.
    
    Based on Definition 4.2 (Page 4):
    RTD = ||INIT(v1) - INIT(v2)||_2 
          + 1/|R|² · (𝟙[r1≠r2] + 𝟙[q1≠q2])
          + w(max(l1,l2)) · OT_RTD(subtrees1, subtrees2)
    """
    r1, tree1 = tree_pair1
    r2, tree2 = tree_pair2
    
    entity1, subtrees1 = tree1
    entity2, subtrees2 = tree2
    
    q1 = subgraph1.query_relation
    q2 = subgraph2.query_relation
    
    # Component 1: Initial embedding difference
    # Handle virtual entities (from blank tree augmentation)
    # Per Definition B.1: blank trees have distinct labels, so compute distance normally
    VIRTUAL_ENTITY = -1
    
    # Get embeddings, using zero vector for virtual entities (distinct from others)
    init1 = subgraph1.initial_embeddings.get(entity1, torch.zeros(20))
    init2 = subgraph2.initial_embeddings.get(entity2, torch.zeros(20))
    embedding_diff = torch.norm(init1 - init2, p=2).item()
    
    # Component 2: Relation and query penalties
    num_relations = max(subgraph1.nrelation, subgraph2.nrelation)
    
    relation_penalty = (1.0 / (num_relations ** 2)) * (
        int(r1 != r2) + int(q1 != q2)
    )
    
    # Component 3: Optimal transport of subtrees
    l1 = _get_tree_depth(tree1)
    l2 = _get_tree_depth(tree2)
    max_depth = max(l1, l2)
    
    weight = _compute_weight_function(max_depth, total_layers=3)
    
    # Augment subtrees if needed
    subtrees1_aug, subtrees2_aug = _blank_tree_augmentation(subtrees1, subtrees2)
    
    subtree_ot = _optimal_transport_rtd(
        subtrees1_aug, subtrees2_aug, subgraph1, subgraph2
    )
    
    rtd = embedding_diff + relation_penalty + weight * subtree_ot
    
    return rtd


def _get_tree_depth(tree: Tuple) -> int:
    """Get depth of a tree."""
    entity, subtrees = tree
    if not subtrees:
        return 0
    max_subtree_depth = max(_get_tree_depth(st[1]) for st in subtrees)
    return 1 + max_subtree_depth


def _compute_weight_function(depth: int, total_layers: int) -> float:
    """
    Compute weight function w(l) for RTD computation.
    
    NOTE: This is an IMPLEMENTATION CHOICE. The paper (Definition 4.2) uses a generic
    weight function w(l) but does not specify an exact closed-form formula. The paper's
    theoretical analysis (Theorem 4.5) gives Lipschitz bounds in terms of layer-wise
    constants, but doesn't prescribe this specific binomial formula.
    
    This implementation uses the binomial coefficient formula assuming history function
    θ(k) = k-1 (common in subgraph reasoning models like GraIL, NBFNet):
        w(l) = (L choose L-l+1) / (L choose L-l)
    
    Alternative formulations exist for different history functions (e.g., θ(k) = 0).
    
    Args:
        depth: Depth l of the tree
        total_layers: Total number of layers L in the model
        
    Returns:
        Weight w(l) for the given depth
    """
    from math import comb
    
    L = total_layers
    if depth >= L:
        return 1.0
    
    numerator = comb(L, L - depth + 1) if L - depth + 1 >= 0 else 1
    denominator = comb(L, L - depth) if L - depth >= 0 else 1
    
    return numerator / denominator if denominator > 0 else 1.0


def _blank_tree_augmentation(trees1: List, trees2: List) -> Tuple[List, List]:
    """
    Augment smaller list with blank trees to match sizes.
    
    Based on Definition B.1 (Page 23):
    Blank tree T_0 = (virtual_entity, [])
    """
    BLANK_RELATION = -2
    VIRTUAL_ENTITY = -1
    BLANK_TREE = (BLANK_RELATION, (VIRTUAL_ENTITY, []))
    
    n1, n2 = len(trees1), len(trees2)
    
    if n1 < n2:
        trees1 = trees1 + [BLANK_TREE] * (n2 - n1)
    elif n2 < n1:
        trees2 = trees2 + [BLANK_TREE] * (n1 - n2)
    
    return trees1, trees2


def _optimal_transport_rtd(
    trees1: List,
    trees2: List,
    subgraph1: Subgraph,
    subgraph2: Subgraph,
    use_exact_ot: bool = False
) -> float:
    """
    Compute optimal transport distance between tree lists using RTD as cost.
    
    Per Appendix E:
    - For inner RTD computations (subtrees): uses Sinkhorn (GeomLoss) for speed
    - For outer RTMD computation (all entities): uses exact OT solver (POT library)
    
    Args:
        trees1: List of (relation, tree) pairs from subgraph1
        trees2: List of (relation, tree) pairs from subgraph2
        subgraph1: First subgraph
        subgraph2: Second subgraph
        use_exact_ot: If True, use exact OT solver; if False, use Sinkhorn
    
    Returns:
        OT distance between tree lists
    """
    n = len(trees1)  # Should equal len(trees2) after augmentation
    
    if n == 0:
        return 0.0
    
    # Build cost matrix: C[i,j] = RTD(trees1[i], trees2[j])
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = _compute_rtd(
                trees1[i], trees2[j], subgraph1, subgraph2
            )
    
    # Uniform distributions
    a = np.ones(n) / n
    b = np.ones(n) / n
    
    if use_exact_ot:
        # Use exact OT solver (Hungarian algorithm / linear programming)
        # This matches the paper's Appendix E for the outer RTMD term
        try:
            import ot  # POT library
            ot_dist = ot.emd2(a, b, cost_matrix)
        except ImportError:
            # Silently fall back to Sinkhorn if POT not available
            ot_dist = _sinkhorn_distance(a, b, cost_matrix, epsilon=0.01)
    else:
        # Use Sinkhorn (entropy-regularized OT) for inner RTD terms
        ot_dist = _sinkhorn_distance(a, b, cost_matrix, epsilon=0.01)
    
    return ot_dist


def _sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float = 0.01,
    max_iter: int = 100
) -> float:
    """
    Sinkhorn algorithm for approximate optimal transport.
    
    From Appendix E: "we use the Sinkhorn algorithm (Cuturi, 2013)"
    """
    # Convert to torch for potential GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.from_numpy(a).float().to(device)
    b = torch.from_numpy(b).float().to(device)
    C = torch.from_numpy(cost_matrix).float().to(device)
    
    # Gibbs kernel
    K = torch.exp(-C / epsilon)
    
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    for _ in range(max_iter):
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)
    
    # Transport plan
    P = torch.diag(u) @ K @ torch.diag(v)
    
    # Distance
    distance = (P * C).sum()
    
    return distance.item()


@torch.no_grad()
def compute_stability_metrics(
    model: nn.Module,
    test_loader,  # DataLoader yielding (pos, cands, mode)
    kg_index,  # KGIndex object
    num_samples: int = 30,
    num_layers: int = 2,
    device: str = 'cuda',
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute empirical Lipschitz constant and stability for a trained model.
    
    Based on Section 6.3: "we compute its empirical Lipschitz constant,
    defined as max_{S1,S2} |f_w(S1) - f_w(S2)| / RTMD(S1,S2)"
    where η̂_f is the empirical Lipschitz constant and C_f = 1/η̂_f is stability.
    
    IMPORTANT LIMITATIONS:
    
    1. MODEL TYPE: This metric is designed for SUBGRAPH REASONING models 
       (e.g., GraIL, NBFNet, RED-GNN) that take subgraph structure as input.
       For standard KGE models (TransE, RotatE, etc.) that only use triplets,
       this metric may not be theoretically meaningful as those models don't
       perform subgraph reasoning.
    
    2. SUBGRAPH SAMPLING: Per paper Section 6.3 and Appendix E, the empirical
       Lipschitz constant should be computed over subgraphs from BOTH:
       - Training KG: S1, S2 ∈ {g(G_tr, e) | e ∈ T_tr}
       - Inference KG: S1, S2 ∈ {g(G_inf, e) | e ∈ T_inf}
       where T_tr and T_inf include BOTH positive and negative triplets.
       
       CURRENT IMPLEMENTATION: Only uses test_loader (likely inference KG only)
       and only positive triplets. This is a SUBSET of what the paper uses.
       For full compliance, pass a combined loader with train+test triplets.
    
    Args:
        model: Trained KGE model (in eval mode)
        test_loader: DataLoader with test triplets (head or tail loader)
                     NOTE: Should ideally include both train+test, pos+neg
        kg_index: KGIndex for subgraph extraction
        num_samples: Number of subgraphs to sample (default: 50)
        num_layers: Depth for RTMD computation (L in paper, default: 1)
        device: Device for computation
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        dict with:
            - stability: C_f = 1/η̂_f (higher is better)
            - num_subgraph_pairs: Number of pairs evaluated
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    model.eval()
    model.to(device)
    
    nentity = model.nentity
    nrelation = model.nrelation
    
    # Step 1: Extract subgraphs and compute scores
    print("Extracting subgraphs and computing scores...")
    extractor = SubgraphExtractor(kg_index, nentity, nrelation, k_hops=2)
    
    subgraphs = []
    scores = []
    
    sample_count = 0
    for pos_samples, cands, mode in test_loader:
        # pos_samples shape: (batch_size, 3) for (h, r, t)
        pos_samples = pos_samples.to(device)
        
        for i in range(pos_samples.shape[0]):
            h, r, t = pos_samples[i].tolist()
            
            # Extract subgraph
            subgraph = extractor.extract(h, r, t)
            
            # Get model score (single mode)
            single_sample = pos_samples[i:i+1]
            score = model(single_sample, mode='single')
            
            subgraphs.append(subgraph)
            scores.append(score.item())
            
            sample_count += 1
            if sample_count >= num_samples:
                break
        
        if sample_count >= num_samples:
            break
    
    print(f"Extracted {len(subgraphs)} subgraphs")
    
    if len(subgraphs) < 2:
        return {
            'lipschitz_constant': float('nan'),
            'stability': float('nan'),
            'num_subgraph_pairs': 0,
        }
    
    # Step 2: Sample pairs and compute Lipschitz constant
    print("Computing empirical Lipschitz constant...")
    
    max_ratio = 0.0
    
    # Sample pairs (limit to avoid O(n²) computation)
    num_pairs_to_sample = min(len(subgraphs) * (len(subgraphs) - 1) // 2, 5000)
    
    # Generate random pairs
    indices = list(range(len(subgraphs)))
    pair_count = 0
    
    for _ in range(num_pairs_to_sample):
        i, j = random.sample(indices, 2)
        
        # Compute RTMD
        rtmd = compute_rtmd(
            subgraphs[i], subgraphs[j],
            num_layers=num_layers
        )
        
        # Avoid division by zero
        if rtmd < 1e-10:
            continue
        
        # Score difference
        score_diff = abs(scores[i] - scores[j])
        
        # Ratio
        ratio = score_diff / rtmd
        max_ratio = max(max_ratio, ratio)
        
        pair_count += 1
        
        # Progress indicator
        if (pair_count) % 500 == 0:
            print(f"  Processed {pair_count}/{num_pairs_to_sample} pairs...")
    
    # Step 3: Compute stability metric
    # Per Section 6.3: η̂_f = max |f(S1) - f(S2)| / RTMD(S1, S2)
    # Stability: C_f = 1/η̂_f (higher is better)
    
    lipschitz_constant = max_ratio
    stability = 1.0 / lipschitz_constant if lipschitz_constant > 0 else float('inf')

    print(f"\nStability (C_f = 1/η̂_f): {stability:.6f}")

    return {
        'stability': stability,
        'num_subgraph_pairs': pair_count,
    }