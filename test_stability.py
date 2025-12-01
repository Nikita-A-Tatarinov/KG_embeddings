"""
Debug script for testing stability metrics computation.

This script extracts subgraphs, computes RTMD, and provides detailed
diagnostics to identify issues in the stability metric computation.

Usage:
    python test_stability.py --ckpt <path> --samples 100 --layers 1
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from dataset.kg_dataset import KGIndex, build_test_loaders
from dataset.utils import load_kg_hf
from eval.stability_metrics import (
    SubgraphExtractor,
    compute_rtmd,
)
from models.registry import create_model


def main():
    parser = argparse.ArgumentParser(description="Test stability metrics with detailed debugging")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", default="WN18RR", help="Dataset name")
    parser.add_argument("--hf-name", default="VLyb/WN18RR", help="HuggingFace dataset name")
    parser.add_argument("--samples", type=int, default=100, help="Number of subgraphs to extract")
    parser.add_argument("--layers", type=int, default=1, help="RTMD tree depth")
    parser.add_argument("--pairs", type=int, default=500, help="Number of pairs to test")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STABILITY METRICS DEBUG TEST")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}")
    print(f"Layers: {args.layers}")
    print(f"Pairs: {args.pairs}")
    print("=" * 80)
    print()
    
    # 1. Load dataset
    print("Step 1: Loading dataset...")
    train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg_hf(args.hf_name)
    
    nentity = len(ent2id)
    nrelation = len(rel2id)
    
    print(f"  Entities: {nentity}")
    print(f"  Relations: {nrelation}")
    print(f"  Test triples: {len(test_ids)}")
    print()
    
    # 2. Build KG index
    print("Step 2: Building KG index...")
    all_triples = torch.cat([train_ids, valid_ids, test_ids], dim=0)
    kg_index = KGIndex(all_triples.tolist(), nentity, nrelation)
    print(f"  Total triples: {len(all_triples)}")
    print()
    
    # 3. Load model
    print("Step 3: Loading model...")
    ckpt_data = torch.load(args.ckpt, map_location="cpu")
    model_config = ckpt_data.get("model_config", {})
    
    base_dim = model_config.get("base_dim", 512)
    gamma = model_config.get("gamma", 12.0)
    
    print(f"  Model: {model_config.get('model_name', 'Unknown')}")
    print(f"  Base dim: {base_dim}")
    print(f"  Gamma: {gamma}")
    
    model = create_model("TransE", nentity=nentity, nrelation=nrelation,
                        base_dim=base_dim, gamma=gamma)
    
    # Attach MI wrapper if needed
    state_dict = ckpt_data.get("model", {})
    if any(key.startswith("_mi") for key in state_dict.keys()):
        print("  Detected MI wrapper, attaching...")
        from models.mi_wrapper import attach_mi
        attach_mi(model, q_dim=None, use_info_nce=True, use_jsd=False)
    
    # Load weights (non-strict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  Ignored {len(unexpected)} unexpected keys")
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    model.to(device)
    model.eval()
    print(f"  Device: {device}")
    print()
    
    # 4. Extract subgraphs
    print("Step 4: Extracting subgraphs...")
    dl_head, _ = build_test_loaders(test_ids, nentity, batch_size=128)
    
    extractor = SubgraphExtractor(kg_index, nentity, nrelation, k_hops=2, max_neighbors=50)
    
    subgraphs = []
    scores = []
    triples = []
    
    with torch.no_grad():
        for pos_samples, _, mode in dl_head:
            pos_samples = pos_samples.to(device)
            
            for i in range(pos_samples.shape[0]):
                if len(subgraphs) >= args.samples:
                    break
                
                h, r, t = pos_samples[i].tolist()
                
                # Extract subgraph
                subgraph = extractor.extract(h, r, t)
                
                # Get model score
                single_sample = pos_samples[i:i+1]
                score = model(single_sample, mode='single')
                
                subgraphs.append(subgraph)
                scores.append(score.item())
                triples.append((h, r, t))
            
            if len(subgraphs) >= args.samples:
                break
    
    print(f"  Extracted: {len(subgraphs)} subgraphs")
    
    # Subgraph statistics
    entity_counts = [len(sg.entities) for sg in subgraphs]
    edge_counts = [len(sg.edges) for sg in subgraphs]
    
    print(f"\n  Subgraph Statistics:")
    print(f"    Entities: min={min(entity_counts)}, max={max(entity_counts)}, "
          f"mean={np.mean(entity_counts):.1f}, median={np.median(entity_counts):.1f}")
    print(f"    Edges: min={min(edge_counts)}, max={max(edge_counts)}, "
          f"mean={np.mean(edge_counts):.1f}, median={np.median(edge_counts):.1f}")
    print()
    
    # 5. Score statistics
    print("Step 5: Analyzing model scores...")
    scores_array = np.array(scores)
    
    print(f"  Score Statistics:")
    print(f"    Min: {scores_array.min():.4f}")
    print(f"    Max: {scores_array.max():.4f}")
    print(f"    Mean: {scores_array.mean():.4f}")
    print(f"    Median: {np.median(scores_array):.4f}")
    print(f"    Std: {scores_array.std():.4f}")
    print(f"    Range: {scores_array.max() - scores_array.min():.4f}")
    print()
    
    # 6. Test RTMD computation
    print(f"Step 6: Computing RTMD for {args.pairs} random pairs...")
    
    rtmd_values = []
    score_diffs = []
    ratios = []
    
    # Track extremes
    min_rtmd = float('inf')
    max_rtmd = 0.0
    min_rtmd_pair = None
    max_rtmd_pair = None
    
    max_ratio = 0.0
    max_ratio_pair = None
    
    zero_rtmd_count = 0
    
    from tqdm import tqdm
    
    indices = list(range(len(subgraphs)))
    
    for _ in tqdm(range(args.pairs), desc="Computing RTMD"):
        i, j = random.sample(indices, 2)
        
        # Compute RTMD
        rtmd = compute_rtmd(subgraphs[i], subgraphs[j], num_layers=args.layers)
        
        # Score difference
        score_diff = abs(scores[i] - scores[j])
        
        # Store
        rtmd_values.append(rtmd)
        score_diffs.append(score_diff)
        
        # Track extremes
        if rtmd < min_rtmd:
            min_rtmd = rtmd
            min_rtmd_pair = (i, j, rtmd, score_diff)
        
        if rtmd > max_rtmd:
            max_rtmd = rtmd
            max_rtmd_pair = (i, j, rtmd, score_diff)
        
        if rtmd < 1e-10:
            zero_rtmd_count += 1
            continue
        
        # Compute ratio
        ratio = score_diff / rtmd
        ratios.append(ratio)
        
        if ratio > max_ratio:
            max_ratio = ratio
            max_ratio_pair = (i, j, rtmd, score_diff, ratio)
    
    print()
    
    # Convert to numpy
    rtmd_array = np.array(rtmd_values)
    score_diff_array = np.array(score_diffs)
    ratio_array = np.array(ratios) if ratios else np.array([0])
    
    # 7. Print detailed statistics
    print("=" * 80)
    print("RTMD STATISTICS")
    print("=" * 80)
    print(f"Count: {len(rtmd_array)}")
    print(f"Min: {rtmd_array.min():.8f}")
    print(f"Max: {rtmd_array.max():.8f}")
    print(f"Mean: {rtmd_array.mean():.8f}")
    print(f"Median: {np.median(rtmd_array):.8f}")
    print(f"Std: {rtmd_array.std():.8f}")
    print(f"10th percentile: {np.percentile(rtmd_array, 10):.8f}")
    print(f"90th percentile: {np.percentile(rtmd_array, 90):.8f}")
    print(f"Zero RTMDs: {zero_rtmd_count}/{len(rtmd_array)}")
    print("=" * 80)
    print()
    
    print("=" * 80)
    print("SCORE DIFFERENCE STATISTICS")
    print("=" * 80)
    print(f"Min: {score_diff_array.min():.6f}")
    print(f"Max: {score_diff_array.max():.6f}")
    print(f"Mean: {score_diff_array.mean():.6f}")
    print(f"Median: {np.median(score_diff_array):.6f}")
    print(f"Std: {score_diff_array.std():.6f}")
    print("=" * 80)
    print()
    
    print("=" * 80)
    print("RATIO (score_diff / RTMD) STATISTICS")
    print("=" * 80)
    if len(ratio_array) > 0:
        print(f"Count: {len(ratio_array)}")
        print(f"Min: {ratio_array.min():.6f}")
        print(f"Max: {ratio_array.max():.6f}")
        print(f"Mean: {ratio_array.mean():.6f}")
        print(f"Median: {np.median(ratio_array):.6f}")
        print(f"Std: {ratio_array.std():.6f}")
        print(f"95th percentile: {np.percentile(ratio_array, 95):.6f}")
        print(f"99th percentile: {np.percentile(ratio_array, 99):.6f}")
    else:
        print("No valid ratios (all RTMDs were zero!)")
    print("=" * 80)
    print()
    
    # 8. Show extreme cases
    print("=" * 80)
    print("EXTREME CASES")
    print("=" * 80)
    
    if min_rtmd_pair:
        i, j, rtmd, score_diff = min_rtmd_pair
        print(f"\nMINIMUM RTMD CASE:")
        print(f"  Pair: subgraph[{i}] vs subgraph[{j}]")
        print(f"  RTMD: {rtmd:.8f}")
        print(f"  Score diff: {score_diff:.6f}")
        print(f"  Ratio: {score_diff / rtmd if rtmd > 1e-10 else float('inf'):.2f}")
        print(f"  Triple[{i}]: {triples[i]}")
        print(f"  Triple[{j}]: {triples[j]}")
        print(f"  Score[{i}]: {scores[i]:.6f}")
        print(f"  Score[{j}]: {scores[j]:.6f}")
        print(f"  Entities[{i}]: {len(subgraphs[i].entities)}")
        print(f"  Entities[{j}]: {len(subgraphs[j].entities)}")
    
    if max_rtmd_pair:
        i, j, rtmd, score_diff = max_rtmd_pair
        print(f"\nMAXIMUM RTMD CASE:")
        print(f"  Pair: subgraph[{i}] vs subgraph[{j}]")
        print(f"  RTMD: {rtmd:.8f}")
        print(f"  Score diff: {score_diff:.6f}")
        print(f"  Ratio: {score_diff / rtmd if rtmd > 1e-10 else float('inf'):.2f}")
        print(f"  Triple[{i}]: {triples[i]}")
        print(f"  Triple[{j}]: {triples[j]}")
    
    if max_ratio_pair:
        i, j, rtmd, score_diff, ratio = max_ratio_pair
        print(f"\nMAXIMUM RATIO CASE (Worst Lipschitz):")
        print(f"  Pair: subgraph[{i}] vs subgraph[{j}]")
        print(f"  RTMD: {rtmd:.8f}")
        print(f"  Score diff: {score_diff:.6f}")
        print(f"  Ratio: {ratio:.2f}")
        print(f"  Triple[{i}]: {triples[i]}")
        print(f"  Triple[{j}]: {triples[j]}")
        print(f"  Score[{i}]: {scores[i]:.6f}")
        print(f"  Score[{j}]: {scores[j]:.6f}")
        print(f"  Entities[{i}]: {len(subgraphs[i].entities)}")
        print(f"  Entities[{j}]: {len(subgraphs[j].entities)}")
    
    print("=" * 80)
    print()
    
    # 9. Diagnostics
    print("=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)
    
    issues_found = False
    
    if zero_rtmd_count > 0:
        print(f"⚠️  WARNING: {zero_rtmd_count} pairs had near-zero RTMD")
        issues_found = True
    
    if rtmd_array.mean() < 0.001:
        print(f"🔥 CRITICAL: RTMD values are VERY small (mean={rtmd_array.mean():.8f})")
        print(f"   → This is likely a bug in RTMD computation")
        print(f"   → Expected mean RTMD: ~1-10")
        print(f"   → Actual mean RTMD: {rtmd_array.mean():.8f} (1000x too small!)")
        issues_found = True
    elif rtmd_array.mean() < 0.1:
        print(f"⚠️  WARNING: RTMD values seem small (mean={rtmd_array.mean():.6f})")
        print(f"   → Expected mean: ~1-10")
        issues_found = True
    
    if max_ratio > 1000:
        print(f"🔥 CRITICAL: Max ratio is extremely high ({max_ratio:.2f})")
        print(f"   → This indicates RTMD is near-zero for some pairs")
        print(f"   → Expected max ratio: ~0.02-0.1")
        print(f"   → Actual max ratio: {max_ratio:.2f}")
        issues_found = True
    
    if not issues_found:
        print("✅ No obvious issues detected!")
        print(f"   Mean RTMD: {rtmd_array.mean():.6f}")
        print(f"   Mean ratio: {ratio_array.mean():.6f}")
        print(f"   Estimated Lipschitz constant: {max_ratio:.6f}")
        print(f"   Estimated Stability: {1.0/max_ratio:.6f}")
    
    print("=" * 80)
    print()
    
    # 10. Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Subgraphs extracted: {len(subgraphs)}")
    print(f"Pairs evaluated: {len(rtmd_values)}")
    print(f"Mean RTMD: {rtmd_array.mean():.8f}")
    print(f"Mean score diff: {score_diff_array.mean():.6f}")
    print(f"Empirical Lipschitz constant: {max_ratio:.6f}")
    print(f"Stability (1/Lipschitz): {1.0/max_ratio if max_ratio > 0 else float('inf'):.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()