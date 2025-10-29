# tests/smoke_test_models.py
import random

import torch

from models import create_model, list_models


def seed_all(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_batches(n_ent, n_rel, B=4, K=6, device="cpu"):
    """Create synthetic batches compatible with forward(sample, mode=...)."""
    # Positive triples (B, 3): (h, r, t)
    pos = torch.stack(
        [
            torch.randint(0, n_ent, (B,)),  # h
            torch.randint(0, n_rel, (B,)),  # r
            torch.randint(0, n_ent, (B,)),  # t
        ],
        dim=1,
    ).long()

    # Candidate matrices for corruption (first column is the positive)
    head_cands = torch.randint(0, n_ent, (B, K)).long()
    head_cands[:, 0] = pos[:, 0]  # ensure gold at index 0

    tail_cands = torch.randint(0, n_ent, (B, K)).long()
    tail_cands[:, 0] = pos[:, 2]  # ensure gold at index 0

    # Assemble samples per mode
    single_sample = pos
    head_batch_sample = (pos, head_cands)  # (tail_part=pos, head_part=cands)
    tail_batch_sample = (pos, tail_cands)  # (head_part=pos, tail_part=cands)

    # Move to device
    single_sample = single_sample.to(device)
    head_batch_sample = (head_batch_sample[0].to(device), head_batch_sample[1].to(device))
    tail_batch_sample = (tail_batch_sample[0].to(device), tail_batch_sample[1].to(device))

    return single_sample, head_batch_sample, tail_batch_sample


def bce_loss_for_candidates(logits):
    """
    logits: (B, K) where column 0 is the positive.
    Make a simple BCE-with-logits loss to test backward.
    """
    B, K = logits.shape
    y = torch.zeros_like(logits)
    y[:, 0] = 1.0
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def run_smoke(device=None):
    seed_all(0)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Small synthetic sizes
    n_ent, n_rel = 50, 10
    B, K = 4, 6

    # Use an even base_dim so RotatEv2 is always valid
    base_dim = 12
    dims_to_test = [2, base_dim // 2, base_dim]

    single_sample, head_sample, tail_sample = build_batches(n_ent, n_rel, B=B, K=K, device=device)

    names = list_models()
    print(f"[models] {', '.join(names)}")

    # Optionally restrict the set while developing:
    # names = ["TransE", "DistMult", "ComplEx", "RotatE", "RotatEv2", "PairRE", "TransH"]

    for name in names:
        print(f"\n=== Testing {name} ===")
        # Some models (PairRE) accept extra kwargs; here we keep it minimal.
        extra = {}
        if name.lower() == "pairre":
            extra["nonneg"] = True  # harmless; toggles softplus on r1, r2

        model = create_model(
            name, nentity=n_ent, nrelation=n_rel, base_dim=base_dim, gamma=12.0, evaluator=None, **extra
        ).to(device)

        model.eval()
        with torch.no_grad():
            # Full-dim checks
            s_single = model(single_sample, mode="single")  # (B, 1)
            s_head = model(head_sample, mode="head-batch")  # (B, K)
            s_tail = model(tail_sample, mode="tail-batch")  # (B, K)

            assert s_single.shape == (B, 1), f"{name} single shape {s_single.shape}"
            assert s_head.shape == (B, K), f"{name} head-batch shape {s_head.shape}"
            assert s_tail.shape == (B, K), f"{name} tail-batch shape {s_tail.shape}"

            for T in (s_single, s_head, s_tail):
                assert torch.isfinite(T).all(), f"{name} produced non-finite values"

            # Croppable checks at multiple dims
            for d in dims_to_test:
                # RotatEv2 requires even d; skip odd if present
                if name.lower() == "rotatev2" and (d % 2 != 0):
                    continue
                sc = model(single_sample, mode="single", crop_dim=d)
                hc = model(head_sample, mode="head-batch", crop_dim=d)
                tc = model(tail_sample, mode="tail-batch", crop_dim=d)
                assert sc.shape == (B, 1)
                assert hc.shape == (B, K)
                assert tc.shape == (B, K)
                assert torch.isfinite(sc).all() and torch.isfinite(hc).all() and torch.isfinite(tc).all(), (
                    f"{name} non-finite at crop_dim={d}"
                )

            valid_dims = [d for d in dims_to_test if not (name.lower() == "rotatev2" and d % 2)]

            out_head = model.scores_for_dims(head_sample, valid_dims, mode="head-batch")
            for d, logits in out_head.items():
                assert logits.shape == (B, K), f"{name} head-batch scores_for_dims shape mismatch at d={d}"

            out_tail = model.scores_for_dims(tail_sample, valid_dims, mode="tail-batch")
            for d, logits in out_tail.items():
                assert logits.shape == (B, K), f"{name} tail-batch scores_for_dims shape mismatch at d={d}"

        # Simple backward test (one step)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        opt.zero_grad()
        # Use croppable setting to test that path too
        use_d = dims_to_test[1] if not (name.lower() == "rotatev2" and dims_to_test[1] % 2) else base_dim
        logits_head = model(head_sample, mode="head-batch", crop_dim=use_d)
        logits_tail = model(tail_sample, mode="tail-batch", crop_dim=use_d)
        loss = bce_loss_for_candidates(logits_head) + bce_loss_for_candidates(logits_tail)
        assert torch.isfinite(loss), f"{name} loss is not finite"
        loss.backward()
        # Ensure some gradients exist
        total_grad_norm = 0.0
        cnt = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += float(p.grad.data.norm().detach().cpu())
                cnt += 1
        assert cnt > 0 and total_grad_norm > 0.0, f"{name} produced zero gradients"
        opt.step()
        print(f"{name} ✓ shapes, cropping, forward/backward OK (loss={float(loss.detach()):.4f})")

    print("\nAll models passed the smoke test.")


if __name__ == "__main__":
    run_smoke()
