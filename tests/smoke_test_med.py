# tests/smoke_test_med.py
import random

import torch

from med.med_wrapper import MEDTrainer
from models import create_model, list_models


def seed_all(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(n_ent, n_rel, B=8, K=16, device="cpu"):
    pos = (
        torch.stack(
            [
                torch.randint(0, n_ent, (B,)),
                torch.randint(0, n_rel, (B,)),
                torch.randint(0, n_ent, (B,)),
            ],
            dim=1,
        )
        .long()
        .to(device)
    )

    head_cands = torch.randint(0, n_ent, (B, K)).long().to(device)
    tail_cands = torch.randint(0, n_ent, (B, K)).long().to(device)
    return pos, head_cands, tail_cands


def run_smoke(device=None):
    seed_all(0)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    n_ent, n_rel = 80, 12
    base_dim = 12  # even to satisfy RotatEv2
    dims = [4, 8, 12]  # sub-models we train mutually
    B, K = 8, 16

    pos, head_cands, tail_cands = make_batch(n_ent, n_rel, B=B, K=K, device=device)

    names = list_models()
    print("[models]", ", ".join(names))

    for name in names:
        print(f"\n=== MED step on {name} ===")
        extra = {}
        if name.lower() == "pairre":
            extra["nonneg"] = True

        model = create_model(
            name, nentity=n_ent, nrelation=n_rel, base_dim=base_dim, gamma=12.0, evaluator=None, **extra
        ).to(device)

        # Filter dims if this model has special constraints (handled inside MEDTrainer)
        med = MEDTrainer(model, d_list=dims, submodels_per_step=3).to(device)
        unique_params = list({id(p): p for p in med.parameters()}.values())
        opt = torch.optim.Adam(unique_params, lr=1e-3)

        # ---- head-batch MED step ----
        model.train()
        med.train()
        opt.zero_grad()
        loss_h, stats_h = med(pos, head_cands, mode="head-batch")
        assert torch.isfinite(loss_h), f"{name} head loss NaN/Inf"
        loss_h.backward()
        opt.step()
        print(
            f"head-batch: loss={stats_h['L_total']:.4f},\
               L_ml={stats_h['L_ml']:.4f}, L_ei={stats_h['L_ei']:.4f}, dims={stats_h['dims']}"
        )

        # ---- tail-batch MED step ----
        opt.zero_grad()
        loss_t, stats_t = med(pos, tail_cands, mode="tail-batch")
        assert torch.isfinite(loss_t), f"{name} tail loss NaN/Inf"
        loss_t.backward()
        opt.step()
        print(
            f"tail-batch: loss={stats_t['L_total']:.4f}, \
                L_ml={stats_t['L_ml']:.4f}, L_ei={stats_t['L_ei']:.4f}, dims={stats_t['dims']}"
        )

        # sanity: parameters moved a bit
        with torch.no_grad():
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.data.norm())
            assert total_norm >= 0.0  # just ensure backward ran

    print("\nAll MED smoke steps completed.")


if __name__ == "__main__":
    run_smoke()
