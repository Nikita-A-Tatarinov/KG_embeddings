# data/kg_dataset.py
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .utils import _maybe_read_id_mapping, _read_triples_id_txt, _read_triples_txt


def load_kg(
    root: str,
    dataset: str,  # "FB15k-237" or "WN18RR"
    prefer_ids: bool = False,  # set True if you have *2id.txt files and want to use them
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, dict[str, int], dict[str, int]]:
    """
    Returns: train_ids, valid_ids, test_ids (Nx3 long), entity2id, relation2id
    Accepts either:
      - raw 'train/valid/test.txt' with string triples
      - or OpenKE-style '*2id.txt' + '*2id.txt' and 'train2id.txt' etc.
    """
    droot = os.path.join(root, dataset)
    if not os.path.isdir(droot):
        raise FileNotFoundError(f"Dataset dir not found: {droot}")

    # Try OpenKE-style first if prefer_ids
    ent2id_path = os.path.join(droot, "entity2id.txt")
    rel2id_path = os.path.join(droot, "relation2id.txt")
    train2id = os.path.join(droot, "train2id.txt")
    valid2id = os.path.join(droot, "valid2id.txt")
    test2id = os.path.join(droot, "test2id.txt")

    has_ids = all(os.path.exists(p) for p in [ent2id_path, rel2id_path, train2id, valid2id, test2id])

    if prefer_ids and has_ids:
        e2id = _maybe_read_id_mapping(ent2id_path) or {}
        r2id = _maybe_read_id_mapping(rel2id_path) or {}
        # read triples in id order
        tr = _read_triples_id_txt(train2id)
        va = _read_triples_id_txt(valid2id)
        te = _read_triples_id_txt(test2id)
        train = torch.tensor(tr, dtype=torch.long)
        valid = torch.tensor(va, dtype=torch.long)
        test = torch.tensor(te, dtype=torch.long)
        return train, valid, test, e2id, r2id

    # Else fall back to raw string triples
    tpath = os.path.join(droot, "train.txt")
    vpath = os.path.join(droot, "valid.txt")
    spath = os.path.join(droot, "test.txt")
    if not all(os.path.exists(p) for p in [tpath, vpath, spath]):
        raise FileNotFoundError(f"Missing raw split files under {droot} (need train.txt / valid.txt / test.txt)")

    train_tr = _read_triples_txt(tpath)
    valid_tr = _read_triples_txt(vpath)
    test_tr = _read_triples_txt(spath)

    # Build maps from *all* splits
    ent2id: dict[str, int] = {}
    rel2id: dict[str, int] = {}

    def _id_for(d: dict[str, int], k: str):
        if k not in d:
            d[k] = len(d)
        return d[k]

    def _encode(triples):
        out = []
        for h, r, t in triples:
            out.append((_id_for(ent2id, h), _id_for(rel2id, r), _id_for(ent2id, t)))
        return torch.tensor(out, dtype=torch.long)

    train_ids = _encode(train_tr)
    valid_ids = _encode(valid_tr)
    test_ids = _encode(test_tr)
    return train_ids, valid_ids, test_ids, ent2id, rel2id


# -----------------------------------
# KG adjacency for filtering / stats
# -----------------------------------
class KGIndex:
    def __init__(self, triples_list: Iterable[Iterable[int]], nentity: int, nrelation: int):
        self.nentity = nentity
        self.nrelation = nrelation
        self.hr2t: dict[tuple[int, int], set] = defaultdict(set)
        self.tr2h: dict[tuple[int, int], set] = defaultdict(set)
        for h, r, t in triples_list:
            self.hr2t[(h, r)].add(t)
            self.tr2h[(t, r)].add(h)


# -----------------------------------
# Train dataset (uniform negatives)
# -----------------------------------
class TrainDataset(Dataset):
    def __init__(
        self,
        triples: torch.LongTensor,  # (N,3) long
        nentity: int,
        nrelation: int,
        negative_size: int = 64,
        mode: str = "head-batch",  # or "tail-batch"
        use_filtered: bool = False,  # filter negs using KGIndex
        all_true: Optional[KGIndex] = None,
        subsampling: bool = True,
    ):
        assert mode in ("head-batch", "tail-batch")
        self.triples = triples
        self.nentity = int(nentity)
        self.nrelation = int(nrelation)
        self.mode = mode
        self.negative_size = int(negative_size)
        self.use_filtered = use_filtered
        self.all_true = all_true
        # Precompute counts for simple subsampling weights
        self.count_hr = defaultdict(int)
        self.count_tr = defaultdict(int)
        for h, r, t in triples.tolist():
            self.count_hr[(h, r)] += 1
            self.count_tr[(t, r)] += 1

    def __len__(self):
        return self.triples.size(0)

    def _sample_negs(self, h, r, t) -> torch.LongTensor:
        K = self.negative_size
        if not self.use_filtered or self.all_true is None:
            # uniform negatives without filtering
            return torch.randint(0, self.nentity, (K,), dtype=torch.long)
        # filtered sampling: avoid generating true triples
        out = []
        tries = 0
        if self.mode == "head-batch":
            banned = self.all_true.tr2h.get((t, r), set())
            while len(out) < K and tries < K * 20:
                x = random.randrange(self.nentity)
                if x not in banned:
                    out.append(x)
                tries += 1
        else:
            banned = self.all_true.hr2t.get((h, r), set())
            while len(out) < K and tries < K * 20:
                x = random.randrange(self.nentity)
                if x not in banned:
                    out.append(x)
                tries += 1
        if len(out) < K:
            # fallback: pad with random (may include banned if exhausted)
            out.extend([random.randrange(self.nentity) for _ in range(K - len(out))])
        return torch.tensor(out, dtype=torch.long)

    def _subsampling_weight(self, h, r, t) -> float:
        # simple 1/sqrt(freq) over (h,r) or (t,r)
        if self.mode == "head-batch":
            c = self.count_tr[(t, r)]
        else:
            c = self.count_hr[(h, r)]
        return 1.0 / (c**0.5)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        neg = self._sample_negs(int(h), int(r), int(t))
        w = self._subsampling_weight(int(h), int(r), int(t))
        pos = torch.tensor([h, r, t], dtype=torch.long)
        return pos, neg, torch.tensor(w, dtype=torch.float), self.mode

    @staticmethod
    def collate_fn(batch):
        pos, neg, w, mode = zip(*batch, strict=True)
        pos = torch.stack(pos, dim=0)  # (B,3)
        neg = torch.stack(neg, dim=0)  # (B,K)
        w = torch.stack(w, dim=0)  # (B,)
        mode = mode[0]  # same for the whole batch
        return pos, neg, w, mode


# -----------------------------------
# Test dataset (full ranking)
# -----------------------------------
class TestDataset(Dataset):
    """
    For evaluation.
    Yields:
      (positive_sample, candidates, mode)
    where 'candidates' is a row-wise list of entity IDs with the **gold entity at column 0**,
    as expected by your evaluator (y_pred_pos=score[:,0], y_pred_neg=score[:,1:]).
    If filtered=True, we try to exclude other true entities; we keep fixed length (= nentity)
    by replacing banned IDs with allowed ones.
    """

    def __init__(
        self,
        triples: torch.LongTensor,  # (N,3)
        nentity: int,
        mode: str = "head-batch",
        filtered: bool = False,
        all_true: Optional[KGIndex] = None,
    ):
        assert mode in ("head-batch", "tail-batch")
        self.triples = triples
        self.nentity = int(nentity)
        self.mode = mode
        self.filtered = filtered
        self.all_true = all_true

    def __len__(self):
        return self.triples.size(0)

    def _row_candidates(self, h: int, r: int, t: int) -> torch.LongTensor:
        n = self.nentity
        # start with sequential 0..n-1
        cands = torch.arange(n, dtype=torch.long)
        gold = h if self.mode == "head-batch" else t

        # put gold at column 0 (swap with current position of gold)
        if gold != 0:
            tmp = int(cands[0].item())
            cands[0] = gold
            cands[gold] = tmp  # since initial is identity, gold sits at index=gold

        if not self.filtered or self.all_true is None:
            return cands

        # filtered: replace banned (excluding gold) with allowed values from the tail of the list
        if self.mode == "head-batch":
            banned = set(self.all_true.tr2h.get((t, r), set()))
            banned.discard(h)
        else:
            banned = set(self.all_true.hr2t.get((h, r), set()))
            banned.discard(t)

        # two-pointer replace: iterate left->right, and swap with allowed from right
        left = 1  # keep index 0 as gold
        right = n - 1
        while left < right:
            lv = int(cands[left].item())
            if lv in banned:
                # find an allowed from the right
                while right > left and int(cands[right].item()) in banned:
                    right -= 1
                if right == left:
                    break
                # swap
                cands[left], cands[right] = cands[right], cands[left]
                right -= 1
            left += 1
        # If still banned remain in the tail, we accept them (rare).
        return cands

    def __getitem__(self, idx):
        h, r, t = self.triples[idx].tolist()
        pos = torch.tensor([h, r, t], dtype=torch.long)
        candidates = self._row_candidates(h, r, t)
        return pos, candidates, self.mode

    @staticmethod
    def collate_fn(batch):
        pos, cands, mode = zip(*batch, strict=True)
        pos = torch.stack(pos, dim=0)  # (B,3)
        cands = torch.stack(cands, dim=0)  # (B, nentity)
        mode = mode[0]
        return pos, cands, mode


# -----------------------------------
# Builders
# -----------------------------------
def build_train_loaders(
    train_ids: torch.LongTensor,
    nentity: int,
    nrelation: int,
    negative_size: int = 64,
    batch_size: int = 1024,
    num_workers: int = 4,
    use_filtered: bool = False,
    all_true: Optional[KGIndex] = None,
):
    ds_head = TrainDataset(train_ids, nentity, nrelation, negative_size, "head-batch", use_filtered, all_true)
    ds_tail = TrainDataset(train_ids, nentity, nrelation, negative_size, "tail-batch", use_filtered, all_true)

    dl_head = DataLoader(
        ds_head,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=TrainDataset.collate_fn,
    )
    dl_tail = DataLoader(
        ds_tail,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=TrainDataset.collate_fn,
    )
    return dl_head, dl_tail


class BidirectionalOneShotIterator:
    """Alternates between head-batch and tail-batch loaders, yielding a tuple
    (positive_sample, negative_sample, subsampling_weight, mode)
    compatible with KGModel.train_step(...).
    """

    def __init__(self, dl_head: DataLoader, dl_tail: DataLoader):
        self.dl_head = iter(dl_head)
        self.dl_tail = iter(dl_tail)
        self.toggle = True
        self._src_head = dl_head
        self._src_tail = dl_tail

    def __next__(self):
        if self.toggle:
            try:
                batch = next(self.dl_head)
            except StopIteration:
                self.dl_head = iter(self._src_head)
                batch = next(self.dl_head)
        else:
            try:
                batch = next(self.dl_tail)
            except StopIteration:
                self.dl_tail = iter(self._src_tail)
                batch = next(self.dl_tail)
        self.toggle = not self.toggle
        return batch


def build_test_loaders(
    split_ids: torch.LongTensor,  # valid or test
    nentity: int,
    batch_size: int = 128,
    num_workers: int = 4,
    filtered: bool = False,
    all_true: Optional[KGIndex] = None,
):
    ds_head = TestDataset(split_ids, nentity, mode="head-batch", filtered=filtered, all_true=all_true)
    ds_tail = TestDataset(split_ids, nentity, mode="tail-batch", filtered=filtered, all_true=all_true)

    dl_head = DataLoader(
        ds_head,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=TestDataset.collate_fn,
    )
    dl_tail = DataLoader(
        ds_tail,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=TestDataset.collate_fn,
    )
    return dl_head, dl_tail
