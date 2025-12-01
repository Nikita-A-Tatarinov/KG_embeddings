"""Simple CLI to evaluate a KGE model on a dataset using filtered MRR/Hits.@

Usage (example):
  PYTHONPATH=. python evaluate.py --model TransE --ckpt path/to/ckpt.pt --data-root dataset --dataset FB15k-237
  PYTHONPATH=. python evaluate.py --model TransE --ckpt path/to/ckpt.pt --dataset FB15k-237 --use-hf
"""

from __future__ import annotations

import argparse

import torch

from dataset.kg_dataset import KGIndex, build_test_loaders, load_kg
from eval.kgc_eval import evaluate_model
from models.registry import create_model
from runner.checkpoint import load_checkpoint
# Add this import at the top
from eval.stability_metrics import compute_stability_metrics

def main_with_stability():
    """Extended evaluation including stability metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (as registered)")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data-root", default=None, help="Path to data root")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--use-hf", action="store_true", help="Load from HuggingFace")
    parser.add_argument("--hf-name", default=None, help="HuggingFace dataset name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--filtered", action="store_true", help="Use filtered evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="Path to save metrics as JSON")
    
    # Stability metrics options
    parser.add_argument("--compute-stability", action="store_true", 
                       help="Compute stability metrics (Lipschitz constant)")
    parser.add_argument("--skip-standard-eval", action="store_true",
                       help="Skip standard MRR/Hits evaluation (only compute stability)")
    parser.add_argument("--stability-samples", type=int, default=500,
                       help="Number of subgraphs to sample for stability")
    parser.add_argument("--stability-layers", type=int, default=3,
                       help="Number of layers for RTMD computation")
    
    args = parser.parse_args()

    # Load data
    if args.use_hf:
        from dataset.utils import load_kg_hf
        hf_dataset_name = args.hf_name or f"KGraph/{args.dataset}"
        print(f"Loading dataset from HuggingFace: {hf_dataset_name}")
        train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg_hf(hf_dataset_name)
    else:
        if args.data_root is None:
            raise ValueError("--data-root required when not using --use-hf")
        print(f"Loading dataset from files: {args.data_root}/{args.dataset}")
        train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg(args.data_root, args.dataset)

    nentity = len(ent2id) if ent2id else int(max(train_ids[:, [0, 2]].max(), valid_ids[:, [0, 2]].max(), test_ids[:, [0, 2]].max())) + 1
    nrelation = len(rel2id) if rel2id else int(max(train_ids[:, 1].max(), valid_ids[:, 1].max(), test_ids[:, 1].max())) + 1

    print(f"Dataset: {args.dataset}")
    print(f"  Entities: {nentity}")
    print(f"  Relations: {nrelation}")
    print(f"  Train: {len(train_ids)} triples")
    print(f"  Valid: {len(valid_ids)} triples")
    print(f"  Test: {len(test_ids)} triples")
    print()

    # Build KGIndex for filtering
    all_true = KGIndex(torch.cat([train_ids, valid_ids, test_ids], dim=0).tolist(), nentity, nrelation)

    # Load checkpoint to infer model architecture
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt_data = torch.load(args.ckpt, map_location="cpu")

    # Try to read model_config from checkpoint (new format)
    model_config = ckpt_data.get("model_config", None)

    if model_config:
        print("Found model_config in checkpoint:")
        print(f"  Model: {model_config['model_name']}")
        print(f"  Base dim: {model_config['base_dim']}")
        print(f"  Entities: {model_config['nentity']}")
        print(f"  Relations: {model_config['nrelation']}")
        print(f"  Gamma: {model_config['gamma']}")
        if model_config.get("med_enabled", False):
            print(f"  MED enabled with dimensions: {model_config.get('d_list', [])}")
        base_dim = model_config["base_dim"]
        gamma = model_config["gamma"]
        med_d_list = model_config.get("d_list", None) if model_config.get("med_enabled", False) else None
    else:
        # Fallback: infer base_dim from checkpoint state_dict (old format)
        print("Warning: model_config not found in checkpoint (old format). Attempting to infer parameters...")
        if "model" in ckpt_data:
            state = ckpt_data["model"]
            if "entity_embedding" in state:
                emb_dim = state["entity_embedding"].shape[1]
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
        gamma = 12.0
        med_d_list = None

    # Create base model with correct sizes
    print(f"\nCreating model: {args.model}")
    model = create_model(args.model, nentity=nentity, nrelation=nrelation, base_dim=base_dim, gamma=gamma)
    
    # Check if checkpoint has MI wrapper parameters
    state_dict = ckpt_data.get("model", {})
    has_mi = any(key.startswith("_mi") for key in state_dict.keys())
    has_rscf = any(key.startswith("_rscf") for key in state_dict.keys())
    
    if has_mi or has_rscf:
        print("\nDetected wrapper modules in checkpoint:")
        if has_mi:
            print("  - MI (Mutual Information) wrapper")
        if has_rscf:
            print("  - RSCF wrapper")
        
        # Attach wrappers to match checkpoint structure
        if has_rscf:
            try:
                from models.rscf_wrapper import attach_rscf
                rscf_module = attach_rscf(model, use_et=True, use_rt=True)
                print("  ✓ Attached RSCF wrapper")
            except Exception as e:
                print(f"  ✗ Failed to attach RSCF: {e}")
        
        if has_mi:
            try:
                from models.mi_wrapper import attach_mi
                mi_module = attach_mi(model, q_dim=None, use_info_nce=True, use_jsd=False)
                print("  ✓ Attached MI wrapper")
            except Exception as e:
                print(f"  ✗ Failed to attach MI: {e}")
    
    # Load checkpoint weights BEFORE wrapping with MEDTrainer
    print("\nLoading checkpoint weights...")
    load_checkpoint(args.ckpt, model, strict=False)
    print("✓ Checkpoint loaded successfully")

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nUsing device: {device}")
    model.to(device)

    # Build test loaders
    dl_head, dl_tail = build_test_loaders(
        test_ids, nentity, batch_size=args.batch_size, 
        filtered=args.filtered, all_true=all_true
    )

    # Check if this was a MED-trained model
    is_med = med_d_list is not None

    # Standard evaluation (can be skipped)
    if not args.skip_standard_eval:
        if is_med:
            # MED model: evaluate each dimension separately
            print(f"\nMED model detected with dimensions: {med_d_list}")
            print("Evaluating each dimension separately...\n")

            all_metrics = {}
            for dim in med_d_list:
                print(f"=== Evaluating dimension {dim} ===")
                metrics_dim = evaluate_model(model, dl_head, dl_tail, device=device, crop_dim=dim)
                all_metrics[f"dim_{dim}"] = metrics_dim

                print(f"Results for dim={dim}:")
                for k, v in metrics_dim.items():
                    print(f"  {k}: {v}")
                print()

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
            print("\n=== Standard Evaluation ===")
            metrics = evaluate_model(model, dl_head, dl_tail, device=device)
            print("Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            
            # Store in all_metrics for consistency
            all_metrics = metrics

            if args.out:
                import json
                import os
                odir = os.path.dirname(args.out)
                if odir:
                    os.makedirs(odir, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(all_metrics, f, indent=2)
                print(f"Saved metrics to {args.out}")
    else:
        print("\n*** Skipping standard evaluation ***")
        all_metrics = {}
    
    # Set eval_model (used for stability computation)
    eval_model = model
    
    # Stability evaluation (optional)
    if args.compute_stability:
        print("\n=== Computing Stability Metrics ===")
        
        # Import stability metrics module
        from eval.stability_metrics import compute_stability_metrics
        
        stability_metrics = compute_stability_metrics(
            eval_model,
            dl_head, 
            all_true,
            num_samples=args.stability_samples,
            num_layers=args.stability_layers,
            device=device
        )
        
        print("\nStability Results:")
        for k, v in stability_metrics.items():
            print(f"  {k}: {v}")
        
        # Save or append stability metrics
        if args.out:
            import json
            import os
            
            # all_metrics is already initialized:
            # - If standard eval ran: contains MRR/Hits metrics
            # - If standard eval skipped: empty dict
            
            # Add stability metrics
            all_metrics['stability'] = stability_metrics
            
            # Save combined metrics
            odir = os.path.dirname(args.out)
            if odir:
                os.makedirs(odir, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\nMetrics saved to {args.out}")

if __name__ == "__main__":
    main_with_stability()