# med_wrapper.py
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MEDTrainer(nn.Module):
    """
    Minimal MED wrapper for any croppable KGModel.
      - Mutual Learning (Huber) between neighbor sub-models
      - Evolutionary Improvement (weighted BCE with logits)
      - Dynamic Loss Weight (alpha grows with dimension)

    Usage:
        med = MEDTrainer(model, d_list=[32,64,128,256])
        loss, stats = med(pos_triples, neg_candidates, mode='head-batch')
        loss.backward(); opt.step()
    """

    def __init__(
        self,
        model,  # a models.KGModel instance
        d_list,
        submodels_per_step: int = 3,
        huber_delta: float = 1.0,
        w1: float = 1.0,  # pos-weight scale
        w2: float = 1.0,  # neg-weight scale
        w3: float = 1.0,  # DLW scale
    ):
        super().__init__()
        self.model = model
        # sanitize & sort dims; also ensure model-specific constraints (e.g., RotatEv2 even d)
        uniq = sorted({int(d) for d in d_list})
        # validate with model's internal checker if available
        checked = []
        for d in uniq:
            try:
                # private but OK within same project
                self.model._check_dim_ok(d)
            except Exception:
                # fallback: keep positive dims only
                if 1 <= d <= getattr(self.model, "max_base_dim", d):
                    checked.append(d)
            else:
                checked.append(d)
        if not checked:
            raise ValueError("d_list is empty or invalid for this model")
        self.d_list = checked
        self.d_max = max(self.d_list)

        self.submodels_per_step = max(2, int(submodels_per_step))
        self.huber_delta = float(huber_delta)
        # learnable scales as in paper defaults
        self.w1 = nn.Parameter(torch.tensor(float(w1)))
        self.w2 = nn.Parameter(torch.tensor(float(w2)))
        self.w3 = nn.Parameter(torch.tensor(float(w3)))

    # ----- utils -----
    @staticmethod
    def _huber(x, delta: float):
        ax = x.abs()
        return torch.where(ax <= delta, 0.5 * ax * ax, delta * (ax - 0.5 * delta))

    def _pick_indices(self):
        n = len(self.d_list)
        m = min(self.submodels_per_step, n)
        if n == m:
            return list(range(n))
        j = random.randrange(n)
        left = max(0, j - m // 2)
        right = min(n, left + m)
        left = max(0, right - m)
        return list(range(left, right))

    # ----- forward: compute MED loss -----
    def forward(self, pos_triples: torch.LongTensor, neg_candidates: torch.LongTensor, mode: str):
        """
        pos_triples: (B,3) LongTensor
        neg_candidates: (B,K) LongTensor of entity ids (head or tail cands depending on `mode`)
        mode: 'head-batch' or 'tail-batch'

        Returns: (loss, stats_dict)
        """
        device = self.model.entity_embedding.device
        dtype = self.model.entity_embedding.dtype

        B = pos_triples.size(0)
        K = neg_candidates.size(1)

        idxs = self._pick_indices()
        dims = [self.d_list[i] for i in idxs]

        # logits at each dim
        s_pos = {}  # d -> (B,)
        s_neg = {}  # d -> (B*K,)
        for d in dims:
            # positive logits (single mode)
            s_p = self.model(pos_triples, mode="single", crop_dim=d).squeeze(1)  # (B,)
            # negative logits (batch mode)
            s_n = self.model((pos_triples, neg_candidates), mode=mode, crop_dim=d)  # (B,K)
            s_pos[d] = s_p
            s_neg[d] = s_n.reshape(B * K)

        # --- Mutual Learning (neighbor Huber on logits) ---
        L_ml = 0.0
        # for a, b in zip(dims[:-1], dims[1:], strict=True):
        #     L_ml = L_ml + self._huber(s_pos[a] - s_pos[b], self.huber_delta).mean()
        #     L_ml = L_ml + self._huber(s_neg[a] - s_neg[b], self.huber_delta).mean()

        for i in range(len(dims) - 1):
            a, b = dims[i], dims[i + 1]
            L_ml = L_ml + self._huber(s_pos[a] - s_pos[b], self.huber_delta).mean()
            L_ml = L_ml + self._huber(s_neg[a] - s_neg[b], self.huber_delta).mean()

        # --- Evolutionary Improvement (weighted BCE) ---
        L_ei = 0.0
        ones_pos = torch.ones(B, device=device, dtype=dtype)
        zeros_neg = torch.zeros(B * K, device=device, dtype=dtype)

        for k, d in enumerate(dims):
            if k == 0:
                # uniform weights
                w_pos = torch.full((B,), 1.0 / B, device=device, dtype=dtype)
                w_neg = torch.full((B * K,), 1.0 / (B * K), device=device, dtype=dtype)
            else:
                prev = dims[k - 1]
                # IMPORTANT: stop grad to previous sub-model logits, but keep w1/w2 learnable
                w_pos = F.softmax(-self.w1 * s_pos[prev].detach(), dim=0)  # harder positives ↑
                w_neg = F.softmax(+self.w2 * s_neg[prev].detach(), dim=0)  # harder negatives ↑

            # Per-example BCE (no 'weight=' arg) so we can multiply by our soft weights
            bce_pos = F.binary_cross_entropy_with_logits(s_pos[d], ones_pos, reduction="none")  # (B,)
            bce_neg = F.binary_cross_entropy_with_logits(s_neg[d], zeros_neg, reduction="none")  # (B*K,)

            # Weighted means (normalize by sum of weights)
            L_pos = (w_pos * bce_pos).sum() / (w_pos.sum() + 1e-12)
            L_neg = (w_neg * bce_neg).sum() / (w_neg.sum() + 1e-12)

            # Dynamic Loss Weight
            d_ratio = d / float(self.d_max)
            alpha = torch.sigmoid(self.w3 * torch.tensor(d_ratio - 0.5, device=device, dtype=dtype))
            L_ei = L_ei + alpha * (L_pos + L_neg)

        loss = L_ei + L_ml
        stats = {
            "L_total": float(loss.detach()),
            "L_ml": float(L_ml.detach()),
            "L_ei": float(L_ei.detach()),
            "dims": dims,
        }
        return loss, stats
