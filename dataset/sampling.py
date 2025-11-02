"""Dataset sampling utilities for faster experimentation.

IMPORTANT: For Knowledge Graph Completion (KGC), we must maintain the full entity/relation
space to ensure valid evaluation. We sample TRIPLES, not entities, and filter evaluation
to only test triples whose entities appear in the training set.
"""

from __future__ import annotations

import random
from typing import Optional

import torch


def sample_kg_dataset(
    train_ids: torch.LongTensor,
    valid_ids: torch.LongTensor,
    test_ids: torch.LongTensor,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
    sample_ratio: Optional[float] = None,
    max_entities: Optional[int] = None,
    max_triples: Optional[int] = None,
    seed: int = 42,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, dict[str, int], dict[str, int], int, int]:
    """
    Sample a subset of a KG dataset for faster experimentation while preserving KGC validity.

    **CRITICAL for KGC**: This function samples TRIPLES, not entities. The full entity/relation
    vocabulary is preserved to ensure evaluation metrics remain valid and comparable.

    Strategies:
    1. sample_ratio: Sample fraction of TRAINING triples, filter valid/test to reachable entities
    2. max_entities: Build training set using top-K connected entities, filter valid/test
    3. max_triples: Sample max N training triples, filter valid/test

    Valid/test filtering ensures we only evaluate on entities that appear in sampled training,
    which is necessary because the model can only learn embeddings for seen entities.

    Args:
        train_ids: Training triples (N, 3)
        valid_ids: Validation triples (M, 3)
        test_ids: Test triples (K, 3)
        ent2id: Entity to ID mapping (PRESERVED - not remapped)
        rel2id: Relation to ID mapping (PRESERVED - not remapped)
        sample_ratio: Fraction of TRAINING triples to keep (e.g., 0.1 for 10%)
        max_entities: Keep top-K most connected entities in training
        max_triples: Maximum number of training triples
        seed: Random seed for reproducibility

    Returns:
        Sampled (train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation)
        Note: nentity and nrelation remain the FULL vocabulary size
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Original vocabulary sizes (NEVER CHANGE THESE)
    nentity = len(ent2id) if ent2id else int(max(train_ids[:, [0, 2]].max(), valid_ids[:, [0, 2]].max(), test_ids[:, [0, 2]].max())) + 1
    nrelation = len(rel2id) if rel2id else int(max(train_ids[:, 1].max(), valid_ids[:, 1].max(), test_ids[:, 1].max())) + 1

    if sample_ratio is not None:
        return _sample_by_ratio_kgc(train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation, sample_ratio)
    elif max_entities is not None:
        return _sample_by_entities_kgc(train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation, max_entities)
    elif max_triples is not None:
        return _sample_by_triples_kgc(train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation, max_triples)
    else:
        # No sampling, return original
        return train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation


def _sample_by_ratio_kgc(train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation, ratio: float):
    """
    Sample training triples by ratio, filter valid/test to reachable entities.
    Preserves full entity/relation vocabulary for valid KGC evaluation.
    """
    ratio = float(ratio)
    assert 0.0 < ratio <= 1.0, f"sample_ratio must be in (0, 1], got {ratio}"

    # Sample TRAINING triples only
    n_train = int(len(train_ids) * ratio)
    train_idx = torch.randperm(len(train_ids))[:n_train]
    train_sampled = train_ids[train_idx]

    # Get entities that appear in sampled training
    train_entities = set(train_sampled[:, 0].tolist() + train_sampled[:, 2].tolist())

    print(f"  Training entities coverage: {len(train_entities)}/{nentity} ({100 * len(train_entities) / nentity:.1f}%)")

    # Filter valid/test to only include triples with entities seen in training
    # (Model can only predict for entities it has learned embeddings for)
    def _filter_to_train_entities(triples):
        mask = torch.tensor([(int(h) in train_entities and int(t) in train_entities) for h, r, t in triples.tolist()], dtype=torch.bool)
        return triples[mask]

    valid_sampled = _filter_to_train_entities(valid_ids)
    test_sampled = _filter_to_train_entities(test_ids)

    print(f"  Valid: {len(valid_ids)} -> {len(valid_sampled)} triples (filtered to training entities)")
    print(f"  Test: {len(test_ids)} -> {len(test_sampled)} triples (filtered to training entities)")

    # Return with ORIGINAL vocabulary size (critical for KGC)
    return train_sampled, valid_sampled, test_sampled, ent2id, rel2id, nentity, nrelation


def _sample_by_entities_kgc(train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation, max_entities: int):
    """
    Sample training using top-K most connected entities.
    Preserves full entity/relation vocabulary for valid KGC evaluation.
    """
    # Count entity degrees in training set
    entity_degrees = {}
    for h, r, t in train_ids.tolist():
        entity_degrees[h] = entity_degrees.get(h, 0) + 1
        entity_degrees[t] = entity_degrees.get(t, 0) + 1

    # Select top-K most connected entities
    sorted_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_entities) <= max_entities:
        # Already small enough, keep everything
        return train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation

    sampled_entities = set([ent for ent, _ in sorted_entities[:max_entities]])

    print(
        f"  Selected top {len(sampled_entities)} most connected entities \
            (avg degree: {sum([deg for _, deg in sorted_entities[:max_entities]]) / len(sampled_entities):.1f})"
    )

    # Filter triples to sampled entities
    def _filter_triples(triples):
        mask = torch.tensor(
            [(int(h) in sampled_entities and int(t) in sampled_entities) for h, r, t in triples.tolist()],
            dtype=torch.bool,
        )
        return triples[mask]

    train_sampled = _filter_triples(train_ids)
    valid_sampled = _filter_triples(valid_ids)
    test_sampled = _filter_triples(test_ids)

    print(f"  Train: {len(train_ids)} -> {len(train_sampled)} triples")
    print(f"  Valid: {len(valid_ids)} -> {len(valid_sampled)} triples")
    print(f"  Test: {len(test_ids)} -> {len(test_sampled)} triples")

    # Return with ORIGINAL vocabulary size (critical for KGC)
    return train_sampled, valid_sampled, test_sampled, ent2id, rel2id, nentity, nrelation


def _sample_by_triples_kgc(train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation, max_triples: int):
    """
    Sample max training triples, filter valid/test to reachable entities.
    Preserves full entity/relation vocabulary for valid KGC evaluation.
    """
    if len(train_ids) <= max_triples:
        # Already small enough
        return train_ids, valid_ids, test_ids, ent2id, rel2id, nentity, nrelation

    # Randomly sample training triples
    train_idx = torch.randperm(len(train_ids))[:max_triples]
    train_sampled = train_ids[train_idx]

    # Get entities in sampled training
    train_entities = set(train_sampled[:, 0].tolist() + train_sampled[:, 2].tolist())

    print(f"  Sampled {len(train_sampled)} training triples")
    print(f"  Training entities coverage: {len(train_entities)}/{nentity} ({100 * len(train_entities) / nentity:.1f}%)")

    # Filter valid/test to training entities
    def _filter_to_train_entities(triples):
        mask = torch.tensor([(int(h) in train_entities and int(t) in train_entities) for h, r, t in triples.tolist()], dtype=torch.bool)
        return triples[mask]

    valid_sampled = _filter_to_train_entities(valid_ids)
    test_sampled = _filter_to_train_entities(test_ids)

    print(f"  Valid: {len(valid_ids)} -> {len(valid_sampled)} triples")
    print(f"  Test: {len(test_ids)} -> {len(test_sampled)} triples")

    # Return with ORIGINAL vocabulary size (critical for KGC)
    return train_sampled, valid_sampled, test_sampled, ent2id, rel2id, nentity, nrelation
