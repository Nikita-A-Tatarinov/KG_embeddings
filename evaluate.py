"""Simple CLI to evaluate a KGE model on a dataset using filtered MRR/Hits.@

Usage (example):
  PYTHONPATH=. python evaluate.py --model TransE --ckpt path/to/ckpt.pt --data-root dataset --dataset FB15k-237
  PYTHONPATH=. python evaluate.py --model TransE --ckpt path/to/ckpt.pt --dataset FB15k-237 --use-hf
"""
from __future__ import annotations

import argparse
import torch

from models.registry import create_model
from runner.checkpoint import load_checkpoint
from dataset.kg_dataset import load_kg, build_test_loaders, KGIndex
from eval.kgc_eval import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name (as registered)")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data-root", default=None,
                        help="Path to data root containing dataset folders (for file-based loading)")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g., FB15k-237, WN18RR)")
    parser.add_argument("--use-hf", action="store_true",
                        help="Load from HuggingFace instead of local files")
    parser.add_argument("--hf-name", default=None,
                        help="HuggingFace dataset name (default: KGraph/<dataset>)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--filtered", action="store_true",
                        help="Use filtered evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None,
                        help="Path to save metrics as JSON")
    args = parser.parse_args()

    # Load triples and maps
    if args.use_hf:
        from dataset.utils import load_kg_hf
        hf_dataset_name = args.hf_name or f"KGraph/{args.dataset}"
        print(f"Loading dataset from HuggingFace: {hf_dataset_name}")
        train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg_hf(
            hf_dataset_name)
    else:
        if args.data_root is None:
            raise ValueError("--data-root is required when not using --use-hf")
        print(f"Loading dataset from files: {args.data_root}/{args.dataset}")
        train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg(
            args.data_root, args.dataset)

    nentity = len(ent2id) if ent2id else int(max(train_ids[:, [0, 2]].max(
    ), valid_ids[:, [0, 2]].max(), test_ids[:, [0, 2]].max())) + 1
    nrelation = len(rel2id) if rel2id else int(
        max(train_ids[:, 1].max(), valid_ids[:, 1].max(), test_ids[:, 1].max())) + 1

    print(f"Dataset: {args.dataset}")
    print(f"  Entities: {nentity}")
    print(f"  Relations: {nrelation}")
    print(f"  Train: {len(train_ids)} triples")
    print(f"  Valid: {len(valid_ids)} triples")
    print(f"  Test: {len(test_ids)} triples")
    print()

    # Build KGIndex for filtering
    all_true = KGIndex(torch.cat(
        [train_ids, valid_ids, test_ids], dim=0).tolist(), nentity, nrelation)

    # Load checkpoint to infer model architecture
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt_data = torch.load(args.ckpt, map_location="cpu")

    # Infer base_dim from checkpoint
    if "model" in ckpt_data:
        state = ckpt_data["model"]
        # entity_embedding shape is (nentity, ENTITY_FACTOR * base_dim)
        # For most models ENTITY_FACTOR=1, but ComplEx has ENTITY_FACTOR=2
        if "entity_embedding" in state:
            emb_dim = state["entity_embedding"].shape[1]
            # Try to infer base_dim (assume ENTITY_FACTOR=1 for RotatE, TransE, etc.)
            # For RotatE: entity_embedding is (nentity, base_dim)
            # For ComplEx: entity_embedding is (nentity, 2*base_dim)
            # We'll use a simple heuristic: check if model name suggests ComplEx
            if args.model.lower() == "complex":
                base_dim = emb_dim // 2
            else:
                base_dim = emb_dim
            print(f"Inferred base_dim from checkpoint: {base_dim}")
        else:
            print("Warning: Could not infer base_dim from checkpoint, using default=100")
            base_dim = 100
    else:
        print("Warning: Could not read checkpoint model state, using default base_dim=100")
        base_dim = 100

    # Create model with correct sizes
    model = create_model(args.model, nentity=nentity,
                         nrelation=nrelation, base_dim=base_dim, gamma=12.0)
    # load checkpoint
    load_checkpoint(args.ckpt, model)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    model.to(device)

    # build test loaders
    dl_head, dl_tail = build_test_loaders(
        test_ids, nentity, batch_size=args.batch_size, filtered=args.filtered, all_true=all_true)

    # Check if model is MED-wrapped (has d_list attribute from MEDTrainer)
    is_med = hasattr(model, 'd_list') and hasattr(model, 'model')

    if is_med:
        # MED model: evaluate each dimension separately
        print(f"MED model detected with dimensions: {model.d_list}")
        print(f"Evaluating each dimension separately...\n")

        all_metrics = {}
        for dim in model.d_list:
            print(f"=== Evaluating dimension {dim} ===")
            # Temporarily crop the base model to this dimension
            original_dim = model.model.base_dim
            model.model.base_dim = dim

            metrics_dim = evaluate_model(
                model.model, dl_head, dl_tail, device=device)
            all_metrics[f"dim_{dim}"] = metrics_dim

            print(f"Results for dim={dim}:")
            for k, v in metrics_dim.items():
                print(f"  {k}: {v}")
            print()

            # Restore original dimension
            model.model.base_dim = original_dim

        # Save per-dimension metrics
        if args.out:
            import json
            import os

            odir = os.path.dirname(args.out)
            if odir:
                os.makedirs(odir, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"Saved per-dimension metrics to {args.out}")
    else:
        # Standard model: single evaluation
        metrics = evaluate_model(model, dl_head, dl_tail, device=device)
        print("Evaluation results:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        if args.out:
            import json
            import os

            odir = os.path.dirname(args.out)
            if odir:
                os.makedirs(odir, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
