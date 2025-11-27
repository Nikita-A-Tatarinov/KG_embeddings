import os
import argparse
import torch
from collections import defaultdict
from tqdm import tqdm
from .utils import load_kg_hf

# Mapping common dataset names to HuggingFace repos
HF_REPOS = {
    "FB15k-237": "KGraph/FB15k-237",
    "WN18RR": "VLyb/WN18RR",
}

def generate_cnf(
    dataset_name: str,
    output_dir: str = "./data",
    precision_threshold: float = 0.01,
    recall_threshold: float = 0.01,
    add_inverse: bool = True
):
    """
    Generates the Context Neighbor Family (CNF) implementing Algorithm 3 (Su et al., 2025).
    """
    # 1. Load Data using your utility
    hf_repo = HF_REPOS.get(dataset_name, dataset_name)
    print(f"Loading {dataset_name} from {hf_repo}...")
    
    # load_kg_hf returns: train_ids, valid_ids, test_ids, ent2id, rel2id
    train_ids, _, _, ent2id, rel2id = load_kg_hf(hf_repo)
    
    # Determine counts
    if rel2id:
        n_relations = len(rel2id)
    else:
        n_relations = train_ids[:, 1].max().item() + 1
        
    print(f"Loaded {len(train_ids)} triples. Found {n_relations} relations.")

    # 2. Augment with Inverse Relations
    # CompGCN and Context Pooling require traversing incoming edges.
    # We map inverse relation r to (r + n_relations).
    triples = train_ids.tolist()
    
    if add_inverse:
        print("Augmenting with inverse relations...")
        inverse_triples = []
        for h, r, t in triples:
            # Inverse: Tail -> (r + n_relations) -> Head
            inverse_triples.append([t, r + n_relations, h])
        triples.extend(inverse_triples)
        total_relations = n_relations * 2
    else:
        total_relations = n_relations

    # 3. Build Adjacency & Count Co-occurrences (Paper Algorithm 3)
    # entity_relations[e] = {r1, r2, ...} set of relations incident to e
    entity_relations = defaultdict(set)
    for h, r, t in triples:
        entity_relations[h].add(r)

    count_r = defaultdict(int)
    count_cooccur = defaultdict(lambda: defaultdict(int))

    print("Counting relation co-occurrences (Algorithm 3)...")
    for entity, relations in tqdm(entity_relations.items(), desc="Processing Entities"):
        rels = list(relations)
        
        # Update individual counts P(r)
        for r in rels:
            count_r[r] += 1
        
        # Update joint counts P(r1, r2)
        # This corresponds to counting N(r_i) in the paper's Algorithm 3
        for i in range(len(rels)):
            for j in range(len(rels)):
                if i == j: continue
                r1 = rels[i]
                r2 = rels[j]
                count_cooccur[r1][r2] += 1

    # 4. Compute Metrics and Build CNF
    # Structure: cnf_data[r_target] = tensor([r_context_1, r_context_2, ...])
    cnf_data = {}
    print(f"Computing logical relevance (P>{precision_threshold}, R>{recall_threshold})...")
    
    valid_contexts_count = 0
    
    for r_target in range(total_relations):
        relevant_neighbors = []
        
        if r_target in count_cooccur:
            potential_contexts = count_cooccur[r_target]
            
            for r_context, co_count in potential_contexts.items():
                # Neighborhood Precision (Eq 2): P(r_target | r_context)
                # "If we see r_context, how likely is r_target also there?"
                prec = co_count / count_r[r_context] if count_r[r_context] > 0 else 0.0

                # Neighborhood Recall (Eq 3): P(r_context | r_target)
                # "If we know r_target is there, how likely do we see r_context?"
                rec = co_count / count_r[r_target] if count_r[r_target] > 0 else 0.0

                # Filter based on thresholds
                if prec > precision_threshold or rec > recall_threshold:
                    relevant_neighbors.append(r_context)
        
        # Store as tensor for efficient loading later
        if relevant_neighbors:
            cnf_data[r_target] = torch.tensor(relevant_neighbors, dtype=torch.long)
            valid_contexts_count += 1
        else:
            # Empty tensor if no context found (prevents key errors in model)
            cnf_data[r_target] = torch.tensor([], dtype=torch.long)

    # 5. Save Artifacts
    # We save to a local folder because the model needs to load this specific file.
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    output_file = os.path.join(save_dir, "cnf.pt")
    
    avg_size = sum(len(v) for v in cnf_data.values()) / total_relations if total_relations > 0 else 0
    print(f"CNF Generation Complete.")
    print(f"  Relations with context: {valid_contexts_count}/{total_relations}")
    print(f"  Avg context size: {avg_size:.2f} relations")
    print(f"Saving to {output_file}...")
    
    torch.save(cnf_data, output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Context Neighbor Family (CNF)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., FB15k-237)")
    parser.add_argument("--output_dir", type=str, default="./data", help="Local root directory to save CNF")
    parser.add_argument("--prec", type=float, default=0.01, help="Precision threshold")
    parser.add_argument("--rec", type=float, default=0.01, help="Recall threshold")
    
    args = parser.parse_args()
    
    # Run generator
    generate_cnf(args.dataset, args.output_dir, args.prec, args.rec)