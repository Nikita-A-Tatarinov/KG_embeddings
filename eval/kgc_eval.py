"""KGC evaluation: filtered MRR and Hits@k utilities."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _score_batch(model, pos_batch: torch.LongTensor, cand_batch: torch.LongTensor, mode: str, device=None, crop_dim=None):
    """Call model in the right mode to get scores for candidates.

    pos_batch: (B,3)
    cand_batch: (B, nentity) with gold at col 0
    crop_dim: optional dimension to crop embeddings (for MED evaluation)
    Returns: scores tensor (B, nentity)
    """
    # Model expects (pos, candidates) sample for head/tail-batch
    sample = (pos_batch.to(device), cand_batch.to(device))
    # model.forward will return logits shaped (B, nentity, 1) or (B, nentity)
    with torch.no_grad():
        # expected shape (B, nentity)
        scores = model(sample, mode=mode, crop_dim=crop_dim)
    # ensure 2D
    if scores.dim() == 3 and scores.size(2) == 1:
        scores = scores.squeeze(2)
    return scores


def evaluate_model(model, dl_head, dl_tail, device=None, hits_ks=(1, 3, 10), crop_dim=None) -> dict:
    """Compute filtered MRR and Hits@k over head and tail loaders.

    crop_dim: optional dimension to crop embeddings (for MED evaluation)
    Returns a dict with keys: mrr, hits@1, hits@3, hits@10, and breakdown for head/tail.
    """
    model.eval()
    device = device or next(model.parameters()).device

    ranks = []  # list of ints (1-based)

    def _process_loader(dl, mode):
        local_ranks = []
        for pos, cands, _mode in dl:
            pos = pos.to(device)
            cands = cands.to(device)
            scores = _score_batch(model, pos, cands, mode,
                                  device=device, crop_dim=crop_dim)
            # gold is at col 0
            # compute rank: higher score = better
            # rank = 1 + (number of scores > gold_score) + 0.5 * (number of scores == gold_score, excluding gold itself)
            gold_scores = scores[:, 0:1]  # (B, 1)
            # Count how many have strictly greater score
            greater = (scores > gold_scores).sum(dim=1)
            # Count how many have equal score (including gold itself, so subtract 1)
            equal = (scores == gold_scores).sum(dim=1) - 1
            # Tie-aware ranking
            rank = (greater + 1 + 0.5 * equal).cpu().tolist()
            local_ranks.extend(rank)
        return local_ranks

    ranks_head = _process_loader(dl_head, mode="head-batch")
    ranks_tail = _process_loader(dl_tail, mode="tail-batch")

    ranks_all = ranks_head + ranks_tail
    ranks_tensor = torch.tensor(ranks_all, dtype=torch.float)
    mrr = torch.mean(1.0 / ranks_tensor).item()
    out = {"mrr": mrr}
    for k in hits_ks:
        out[f"hits@{k}"] = float((ranks_tensor <=
                                 float(k)).float().mean().item())

    out["n_examples"] = int(ranks_tensor.numel())
    out["n_head"] = len(ranks_head)
    out["n_tail"] = len(ranks_tail)
    return out
