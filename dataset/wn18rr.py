# dataloader/wn18rr.py
from __future__ import annotations

from .kg_dataset import BidirectionalOneShotIterator, KGIndex, build_test_loaders, build_train_loaders, load_kg
from .utils import load_kg_hf


def prepare_wn18rr(
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
):
    if use_hf is None:
        use_hf = False

    if use_hf:
        train, valid, test, ent2id, rel2id = load_kg_hf(hf_name or "KGraph/FB15k-237", revision=hf_revision)
    else:
        train, valid, test, ent2id, rel2id = load_kg(root, "WN18RR", prefer_ids=prefer_ids)
    nentity = len(ent2id) if ent2id else int(train[:, [0, 2]].max().item()) + 1
    nrelation = len(rel2id) if rel2id else int(train[:, 1].max().item()) + 1

    all_true = KGIndex(
        triples_list=train.tolist() + valid.tolist() + test.tolist(), nentity=nentity, nrelation=nrelation
    )

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

    v_head, v_tail = build_test_loaders(
        valid, nentity, batch_size=test_bs, num_workers=num_workers, filtered=filtered_eval, all_true=all_true
    )
    t_head, t_tail = build_test_loaders(
        test, nentity, batch_size=test_bs, num_workers=num_workers, filtered=filtered_eval, all_true=all_true
    )

    meta = dict(nentity=nentity, nrelation=nrelation, ent2id=ent2id, rel2id=rel2id)
    return train_iter, (v_head, v_tail), (t_head, t_tail), meta
