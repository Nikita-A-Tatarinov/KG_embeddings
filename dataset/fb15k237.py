# dataloader/fb15k237.py
from __future__ import annotations

from typing import Optional

from .kg_dataset import (
    BidirectionalOneShotIterator,
    KGIndex,
    build_test_loaders,
    build_train_loaders,
    load_kg,
)
from .sampling import sample_kg_dataset
from .utils import load_kg_hf


def prepare_fb15k237(
    root: str,
    use_hf: bool = None,
    hf_name: str = None,
    hf_revision: str = None,
    prefer_ids: bool = False,
    neg_size: int = 64,
    train_bs: int = 1024,
    test_bs: int = 128,
    num_workers: int = 4,
    filtered_eval: bool = True,
    # Sampling parameters for faster experimentation
    sample_ratio: Optional[float] = None,
    max_entities: Optional[int] = None,
    max_triples: Optional[int] = None,
    sample_seed: int = 42,
    # Additional validation sampling for even faster validation during training
    sample_valid_ratio: Optional[float] = None,
):
    if use_hf is None:
        use_hf = False

    if use_hf:
        train, valid, test, ent2id, rel2id = load_kg_hf(
            hf_name or "KGraph/FB15k-237", revision=hf_revision)
    else:
        train, valid, test, ent2id, rel2id = load_kg(
            root, "FB15k-237", prefer_ids=prefer_ids)

    # Apply sampling if requested
    if sample_ratio is not None or max_entities is not None or max_triples is not None:
        print(f"\n=== Sampling FB15k-237 Dataset ===")
        print(
            f"Original sizes: train={len(train)}, valid={len(valid)}, test={len(test)}")
        train, valid, test, ent2id, rel2id, nentity, nrelation = sample_kg_dataset(
            train, valid, test, ent2id, rel2id,
            sample_ratio=sample_ratio,
            max_entities=max_entities,
            max_triples=max_triples,
            seed=sample_seed,
        )
        print(
            f"Sampled sizes: train={len(train)}, valid={len(valid)}, test={len(test)}")
        print(f"Entities: {nentity}, Relations: {nrelation}")
        print("=" * 35 + "\n")
    else:
        nentity = len(ent2id) if ent2id else int(
            train[:, [0, 2]].max().item()) + 1
        nrelation = len(rel2id) if rel2id else int(
            train[:, 1].max().item()) + 1

    # Build all-true index across all splits for filtered sampling/eval
    all_true = KGIndex(
        triples_list=train.tolist() + valid.tolist() + test.tolist(), nentity=nentity, nrelation=nrelation
    )

    # Train loaders + alternating iterator
    dl_h, dl_t = build_train_loaders(
        train,
        nentity,
        nrelation,
        negative_size=neg_size,
        batch_size=train_bs,
        num_workers=num_workers,
        use_filtered=False,
        all_true=all_true,
    )
    train_iter = BidirectionalOneShotIterator(dl_h, dl_t)

    # Eval loaders (valid/test), full ranking; filtered per flag
    # Additional validation sampling for faster training
    if sample_valid_ratio is not None and sample_valid_ratio < 1.0:
        import torch
        n_valid = int(len(valid) * sample_valid_ratio)
        valid_idx = torch.randperm(len(valid))[:n_valid]
        valid_sampled = valid[valid_idx]
        print(
            f"  Additional validation sampling: {len(valid)} → {len(valid_sampled)} triples ({sample_valid_ratio*100:.0f}%)")
        valid_for_loader = valid_sampled
    else:
        valid_for_loader = valid

    v_head, v_tail = build_test_loaders(
        valid_for_loader, nentity, batch_size=test_bs, num_workers=num_workers, filtered=filtered_eval, all_true=all_true
    )
    t_head, t_tail = build_test_loaders(
        test, nentity, batch_size=test_bs, num_workers=num_workers, filtered=filtered_eval, all_true=all_true
    )

    meta = dict(nentity=nentity, nrelation=nrelation,
                ent2id=ent2id, rel2id=rel2id)
    return train_iter, (v_head, v_tail), (t_head, t_tail), meta
