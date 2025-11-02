"""Test MED, RSCF, and MI wrappers with evaluation integration (CPU-friendly).

This test verifies that:
1. MED wrapper works with training and evaluation
2. RSCF wrapper works with training and evaluation
3. MI wrapper works with training and evaluation
4. All wrappers can save/load checkpoints correctly
"""

from __future__ import annotations

import random
import tempfile

import torch

from dataset.kg_dataset import BidirectionalOneShotIterator, KGIndex, build_test_loaders, build_train_loaders
from eval.kgc_eval import evaluate_model
from med.med_wrapper import MEDTrainer
from models.registry import create_model
from runner.checkpoint import load_checkpoint, save_checkpoint


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def create_tiny_kg():
    """Create minimal in-memory KG for testing."""
    # 4 entities, 2 relations, 6 triples
    train_ids = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 2],
            [2, 1, 3],
            [0, 1, 3],
        ],
        dtype=torch.long,
    )

    valid_ids = torch.tensor(
        [
            [1, 1, 2],
        ],
        dtype=torch.long,
    )

    test_ids = torch.tensor(
        [
            [2, 0, 0],
        ],
        dtype=torch.long,
    )

    nentity = 4
    nrelation = 2

    all_true = KGIndex(
        torch.cat([train_ids, valid_ids, test_ids], dim=0).tolist(),
        nentity,
        nrelation,
    )

    return train_ids, valid_ids, test_ids, nentity, nrelation, all_true


def test_rscf_wrapper(device="cpu"):
    """Test RSCF wrapper with training and evaluation."""
    print("\n" + "=" * 60)
    print("Testing RSCF Wrapper")
    print("=" * 60)

    seed_all(42)
    train_ids, valid_ids, test_ids, nentity, nrelation, all_true = create_tiny_kg()

    # Create model with RSCF
    model = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)

    from models.rscf_wrapper import attach_rscf

    attach_rscf(model, use_et=True, use_rt=True)

    model = model.to(device)
    print("✓ RSCF attached to model")

    # Train for 2 steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dl_head, dl_tail = build_train_loaders(
        train_ids,
        nentity,
        nrelation,
        negative_size=3,
        batch_size=2,
        num_workers=0,
        use_filtered=True,
        all_true=all_true,
    )
    train_iter = BidirectionalOneShotIterator(dl_head, dl_tail)

    model.train()
    for step in range(2):
        pos, neg, weight, mode = next(train_iter)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()
        negative_score = model((pos, neg), mode=mode)
        positive_score = model(pos)

        pos_loss = -torch.nn.functional.logsigmoid(positive_score).squeeze(1).mean()
        neg_loss = -torch.nn.functional.logsigmoid(-negative_score).mean()
        loss = (pos_loss + neg_loss) / 2.0

        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: loss={loss.item():.4f}")

    # Evaluate
    model.eval()
    test_dl_head, test_dl_tail = build_test_loaders(test_ids, nentity, batch_size=2, filtered=True, all_true=all_true)
    metrics = evaluate_model(model, test_dl_head, test_dl_tail, device=device)
    print(f"✓ Evaluation: MRR={metrics['mrr']:.4f}")

    # Test checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(tmpdir, "rscf_test", {"step": 2}, model, optimizer)
        model_new = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)
        attach_rscf(model_new, use_et=True, use_rt=True)
        load_checkpoint(ckpt_path, model_new)
        print("✓ Checkpoint save/load works")

    print("✅ RSCF wrapper test passed!\n")


def test_mi_wrapper(device="cpu"):
    """Test MI wrapper with training and evaluation."""
    print("\n" + "=" * 60)
    print("Testing MI Wrapper")
    print("=" * 60)

    seed_all(42)
    train_ids, valid_ids, test_ids, nentity, nrelation, all_true = create_tiny_kg()

    # Create model with MI
    model = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)

    from models.mi_wrapper import attach_mi

    attach_mi(model, use_info_nce=True, use_jsd=False)

    model = model.to(device)
    print("✓ MI attached to model")

    # Train for 2 steps with MI loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dl_head, dl_tail = build_train_loaders(
        train_ids,
        nentity,
        nrelation,
        negative_size=3,
        batch_size=2,
        num_workers=0,
        use_filtered=True,
        all_true=all_true,
    )
    train_iter = BidirectionalOneShotIterator(dl_head, dl_tail)

    model.train()
    for step in range(2):
        pos, neg, weight, mode = next(train_iter)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()

        # Standard KGE loss
        negative_score = model((pos, neg), mode=mode)
        positive_score = model(pos)
        pos_loss = -torch.nn.functional.logsigmoid(positive_score).squeeze(1).mean()
        neg_loss = -torch.nn.functional.logsigmoid(-negative_score).mean()
        kge_loss = (pos_loss + neg_loss) / 2.0

        # MI loss
        mi_loss = model.compute_mi_loss(pos, neg_size=3)

        # Combined loss
        loss = kge_loss + 0.1 * mi_loss

        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: kge_loss={kge_loss.item():.4f}, mi_loss={mi_loss.item():.4f}")

    # Evaluate
    model.eval()
    test_dl_head, test_dl_tail = build_test_loaders(test_ids, nentity, batch_size=2, filtered=True, all_true=all_true)
    metrics = evaluate_model(model, test_dl_head, test_dl_tail, device=device)
    print(f"✓ Evaluation: MRR={metrics['mrr']:.4f}")

    # Test checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(tmpdir, "mi_test", {"step": 2}, model, optimizer)
        model_new = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)
        attach_mi(model_new, use_info_nce=True, use_jsd=False)
        load_checkpoint(ckpt_path, model_new)
        print("✓ Checkpoint save/load works")

    print("✅ MI wrapper test passed!\n")


def test_med_wrapper(device="cpu"):
    """Test MED wrapper with training and evaluation."""
    print("\n" + "=" * 60)
    print("Testing MED Wrapper")
    print("=" * 60)

    seed_all(42)
    train_ids, valid_ids, test_ids, nentity, nrelation, all_true = create_tiny_kg()

    # Create model with MED
    base_model = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)
    model = MEDTrainer(base_model, d_list=[2, 4, 8], submodels_per_step=2)
    model = model.to(device)
    print("✓ MED wrapper created")

    # Train for 2 steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dl_head, dl_tail = build_train_loaders(
        train_ids,
        nentity,
        nrelation,
        negative_size=3,
        batch_size=2,
        num_workers=0,
        use_filtered=True,
        all_true=all_true,
    )
    train_iter = BidirectionalOneShotIterator(dl_head, dl_tail)

    model.train()
    for step in range(2):
        pos, neg, weight, mode = next(train_iter)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()
        loss, stats = model(pos, neg, mode=mode)
        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: L_total={stats['L_total']:.4f}, L_ml={stats['L_ml']:.4f}, L_ei={stats['L_ei']:.4f}")

    # Evaluate (use underlying model)
    model.eval()
    test_dl_head, test_dl_tail = build_test_loaders(test_ids, nentity, batch_size=2, filtered=True, all_true=all_true)
    metrics = evaluate_model(model.model, test_dl_head, test_dl_tail, device=device)
    print(f"✓ Evaluation: MRR={metrics['mrr']:.4f}")

    # Test checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(tmpdir, "med_test", {"step": 2}, model, optimizer)
        base_model_new = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)
        model_new = MEDTrainer(base_model_new, d_list=[2, 4, 8], submodels_per_step=2)
        load_checkpoint(ckpt_path, model_new)
        print("✓ Checkpoint save/load works")

    print("✅ MED wrapper test passed!\n")


def test_combined_wrappers(device="cpu"):
    """Test MED + MI combination (advanced scenario)."""
    print("\n" + "=" * 60)
    print("Testing MED + MI Combined")
    print("=" * 60)

    seed_all(42)
    train_ids, valid_ids, test_ids, nentity, nrelation, all_true = create_tiny_kg()

    # Create model with MI, then wrap in MED
    base_model = create_model("TransE", nentity=nentity, nrelation=nrelation, base_dim=8, gamma=12.0)

    from models.mi_wrapper import attach_mi

    attach_mi(base_model, use_info_nce=True, use_jsd=False)

    model = MEDTrainer(base_model, d_list=[4, 8], submodels_per_step=2, mi_lambda=0.1, mi_neg_size=3)
    model = model.to(device)
    print("✓ MED + MI combined")

    # Train for 2 steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dl_head, dl_tail = build_train_loaders(
        train_ids,
        nentity,
        nrelation,
        negative_size=3,
        batch_size=2,
        num_workers=0,
        use_filtered=True,
        all_true=all_true,
    )
    train_iter = BidirectionalOneShotIterator(dl_head, dl_tail)

    model.train()
    for step in range(2):
        pos, neg, weight, mode = next(train_iter)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()
        loss, stats = model(pos, neg, mode=mode)
        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: L_total={stats['L_total']:.4f}")

    # Evaluate
    model.eval()
    test_dl_head, test_dl_tail = build_test_loaders(test_ids, nentity, batch_size=2, filtered=True, all_true=all_true)
    metrics = evaluate_model(model.model, test_dl_head, test_dl_tail, device=device)
    print(f"✓ Evaluation: MRR={metrics['mrr']:.4f}")

    print("✅ MED + MI combination test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("WRAPPER INTEGRATION TESTS (CPU-friendly)")
    print("=" * 60)

    device = "cpu"

    test_rscf_wrapper(device)
    test_mi_wrapper(device)
    test_med_wrapper(device)
    test_combined_wrappers(device)

    print("\n" + "=" * 60)
    print("✅ ALL WRAPPER INTEGRATION TESTS PASSED!")
    print("=" * 60)
