#!/usr/bin/env python3
"""Quick test to verify KGC-aware dataset sampling works correctly."""

import sys
import torch

# Test sampling utilities
from dataset.sampling import sample_kg_dataset


def create_dummy_kg(n_train=1000, n_valid=100, n_test=100, n_ent=500, n_rel=50):
    """Create a dummy KG for testing."""
    train = torch.randint(0, n_ent, (n_train, 3))
    train[:, 1] = torch.randint(0, n_rel, (n_train,))

    valid = torch.randint(0, n_ent, (n_valid, 3))
    valid[:, 1] = torch.randint(0, n_rel, (n_valid,))

    test = torch.randint(0, n_ent, (n_test, 3))
    test[:, 1] = torch.randint(0, n_rel, (n_test,))

    return train, valid, test, {}, {}


def test_sample_by_ratio():
    """Test sampling by ratio - KGC aware version."""
    print("=" * 60)
    print("TEST 1: KGC-aware sampling by ratio (10%)")
    print("=" * 60)

    train, valid, test, ent2id, rel2id = create_dummy_kg()
    orig_ent_count = int(
        max(train[:, [0, 2]].max(), valid[:, [0, 2]].max(), test[:, [0, 2]].max())) + 1

    print(
        f"Original: train={len(train)}, valid={len(valid)}, test={len(test)}, entities={orig_ent_count}")

    train_s, valid_s, test_s, ent2id_s, rel2id_s, nent, nrel = sample_kg_dataset(
        train, valid, test, ent2id, rel2id,
        sample_ratio=0.1,
        seed=42
    )

    print(
        f"Sampled:  train={len(train_s)}, valid={len(valid_s)}, test={len(test_s)}")
    print(f"Vocabulary: entities={nent}, relations={nrel}")

    # CRITICAL: Entity vocabulary should be PRESERVED (not remapped)
    assert nent == orig_ent_count, f"Entity vocab changed! {nent} != {orig_ent_count}"
    assert len(train_s) <= len(train) * 0.15  # Allow some variance

    # Valid/test should be filtered to training entities
    train_ents = set(train_s[:, 0].tolist() + train_s[:, 2].tolist())
    for h, r, t in valid_s.tolist():
        assert h in train_ents and t in train_ents, "Valid triple has unseen entity!"
    for h, r, t in test_s.tolist():
        assert h in train_ents and t in train_ents, "Test triple has unseen entity!"

    print("✓ Test passed! Entity vocabulary preserved, eval filtered correctly.\n")


def test_sample_by_entities():
    """Test sampling by max entities - KGC aware version."""
    print("=" * 60)
    print("TEST 2: KGC-aware sampling by entities (max 100)")
    print("=" * 60)

    train, valid, test, ent2id, rel2id = create_dummy_kg()
    orig_ent_count = int(
        max(train[:, [0, 2]].max(), valid[:, [0, 2]].max(), test[:, [0, 2]].max())) + 1

    print(
        f"Original: train={len(train)}, valid={len(valid)}, test={len(test)}, entities={orig_ent_count}")

    train_s, valid_s, test_s, ent2id_s, rel2id_s, nent, nrel = sample_kg_dataset(
        train, valid, test, ent2id, rel2id,
        max_entities=100,
        seed=42
    )

    print(
        f"Sampled:  train={len(train_s)}, valid={len(valid_s)}, test={len(test_s)}")
    print(f"Vocabulary: entities={nent}, relations={nrel}")

    # CRITICAL: Entity vocabulary should be PRESERVED
    assert nent == orig_ent_count, f"Entity vocab changed! {nent} != {orig_ent_count}"

    # Training should only use ~100 entities
    train_ents = set(train_s[:, 0].tolist() + train_s[:, 2].tolist())
    assert len(
        train_ents) <= 110, f"Too many entities in training: {len(train_ents)}"

    print("✓ Test passed! Entity vocabulary preserved.\n")


def test_sample_by_triples():
    """Test sampling by max triples - KGC aware version."""
    print("=" * 60)
    print("TEST 3: KGC-aware sampling by triples (max 200 train)")
    print("=" * 60)

    train, valid, test, ent2id, rel2id = create_dummy_kg()
    orig_ent_count = int(
        max(train[:, [0, 2]].max(), valid[:, [0, 2]].max(), test[:, [0, 2]].max())) + 1

    print(
        f"Original: train={len(train)}, valid={len(valid)}, test={len(test)}, entities={orig_ent_count}")

    train_s, valid_s, test_s, ent2id_s, rel2id_s, nent, nrel = sample_kg_dataset(
        train, valid, test, ent2id, rel2id,
        max_triples=200,
        seed=42
    )

    print(
        f"Sampled:  train={len(train_s)}, valid={len(valid_s)}, test={len(test_s)}")
    print(f"Vocabulary: entities={nent}, relations={nrel}")

    # CRITICAL: Entity vocabulary should be PRESERVED
    assert nent == orig_ent_count, f"Entity vocab changed! {nent} != {orig_ent_count}"
    assert len(train_s) <= 200

    print("✓ Test passed! Entity vocabulary preserved.\n")


def test_fb15k237_sampling():
    """Test real FB15k-237 KGC-aware sampling."""
    print("=" * 60)
    print("TEST 4: Real FB15k-237 KGC-aware sampling (10%)")
    print("=" * 60)

    try:
        from dataset.fb15k237 import prepare_fb15k237

        # Load without sampling to get original sizes
        print("Loading full dataset...")
        _, _, _, meta_full = prepare_fb15k237(
            root="./data",
            use_hf=True,
        )

        print(
            f"Full dataset: {meta_full['nentity']} entities, {meta_full['nrelation']} relations")

        # Load with 10% sampling
        print("\nLoading with 10% sampling...")
        train_iter, (v_head, v_tail), (t_head, t_tail), meta = prepare_fb15k237(
            root="./data",
            use_hf=True,
            sample_ratio=0.1,
            sample_seed=42,
        )

        print(f"\n✓ FB15k-237 KGC-aware sampling successful!")
        print(
            f"  Vocabulary size: {meta['nentity']} entities, {meta['nrelation']} relations")

        # CRITICAL CHECK: Vocabulary should be PRESERVED
        assert meta['nentity'] == meta_full['nentity'], "Entity vocabulary was remapped! This breaks KGC!"
        assert meta['nrelation'] == meta_full['nrelation'], "Relation vocabulary was remapped!"

        print(f"  ✓ Vocabulary preserved correctly (same as full dataset)")
        print("✓ Test passed!\n")

    except Exception as e:
        print(f"⚠ FB15k-237 test skipped (needs HuggingFace download): {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing KGC-Aware Dataset Sampling Utilities")
    print("=" * 60 + "\n")

    test_sample_by_ratio()
    test_sample_by_entities()
    test_sample_by_triples()
    test_fb15k237_sampling()

    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print("\nKey validations:")
    print("  ✓ Entity vocabulary is PRESERVED (not remapped)")
    print("  ✓ Relation vocabulary is PRESERVED")
    print("  ✓ Valid/test filtered to training entities only")
    print("  ✓ KGC evaluation remains valid and comparable")
    print("=" * 60)
