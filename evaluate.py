"""Simple CLI to evaluate a KGE model on a dataset using filtered MRR/Hits.@

Usage (example):
  PYTHONPATH=. python evaluate.py --model TransE --ckpt path/to/ckpt.pt --data-root dataset --dataset FB15k-237
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
    parser.add_argument("--data-root", required=True,
                        help="Path to data root containing dataset folders")
    parser.add_argument("--dataset", required=True,
                        help="Dataset folder name (e.g., FB15k-237)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--filtered", action="store_true",
                        help="Use filtered evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None,
                        help="Path to save metrics as JSON")
    args = parser.parse_args()

    # Load triples and maps
    train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg(
        args.data_root, args.dataset)

    nentity = len(ent2id)
    nrelation = len(rel2id)

    # Build KGIndex for filtering
    all_true = KGIndex(torch.cat(
        [train_ids, valid_ids, test_ids], dim=0).tolist(), nentity, nrelation)

    # Create model stub with correct sizes. We assume config defaults are acceptable; user may modify.
    model = create_model(args.model, nentity=nentity,
                         nrelation=nrelation, base_dim=100, gamma=12.0)
    # load checkpoint
    load_checkpoint(args.ckpt, model)
    device = torch.device(args.device)
    model.to(device)

    # build test loaders
    dl_head, dl_tail = build_test_loaders(
        test_ids, nentity, batch_size=args.batch_size, filtered=args.filtered, all_true=all_true)

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
