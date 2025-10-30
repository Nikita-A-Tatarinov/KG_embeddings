# models/rscf.py
from __future__ import annotations

import torch
import torch.nn as nn


class RSCFModule(nn.Module):
    """
    Implements the core Relation-Semantics Consistent Filter (RSCF) transformations.

    - Entity transformation (ET): maps a relation embedding -> change vector, normalizes it
      and applies a rooted transform: e_r = (N(r A1) + 1) ⊗ e
    - Relation transformation (RT): maps head/tail entity -> relation-specific modifiers and
      computes r_ht = (N(h A2) + 1) ⊗ (N(t A3) + 1) ⊗ r

    The module uses three shared affine matrices A1, A2, A3 (learned).
    """

    def __init__(self, e_dim: int, r_dim: int, p: int = 2, eps: float = 1e-12):
        super().__init__()
        self.e_dim = int(e_dim)
        self.r_dim = int(r_dim)
        self.p = int(p)
        self.eps = float(eps)

        # A1: maps relation_dim -> entity_dim  (r @ A1 -> e_dim)
        self.A1 = nn.Parameter(torch.empty(self.r_dim, self.e_dim))
        # A2: maps entity_dim -> relation_dim  (h @ A2 -> r_dim)
        self.A2 = nn.Parameter(torch.empty(self.e_dim, self.r_dim))
        # A3: maps entity_dim -> relation_dim  (t @ A3 -> r_dim)
        self.A3 = nn.Parameter(torch.empty(self.e_dim, self.r_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier-ish init for affine matrices
        nn.init.xavier_uniform_(self.A1)
        nn.init.xavier_uniform_(self.A2)
        nn.init.xavier_uniform_(self.A3)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        denom = x.norm(p=self.p, dim=-1, keepdim=True).clamp_min(self.eps)
        return x / denom

    def transform_entity(self, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        e: (..., e_dim)
        r: (..., r_dim)
        returns e_r: (..., e_dim)
        """
        # compute change = r @ A1 -> (..., e_dim)
        # Support cropped embeddings: slice A1 to match current r/e trailing dims
        r_cur = r.size(-1)
        e_cur = e.size(-1)
        A1 = self.A1[:r_cur, :e_cur]
        change = torch.matmul(r, A1)
        change_n = self._normalize(change)
        # rooted transform: (change_n + 1) ⊗ e
        return (change_n + 1.0) * e

    def transform_relation(self, h: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        h: (..., e_dim), t: (..., e_dim), r: (..., r_dim)
        returns r_ht: (..., r_dim)
        """
        # Support cropped embeddings: slice A2/A3 to match current dims
        e_cur = h.size(-1)
        r_cur = r.size(-1)
        A2 = self.A2[:e_cur, :r_cur]
        A3 = self.A3[:e_cur, :r_cur]
        h_feat = torch.matmul(h, A2)
        t_feat = torch.matmul(t, A3)
        h_n = self._normalize(h_feat)
        t_n = self._normalize(t_feat)
        return (h_n + 1.0) * (t_n + 1.0) * r

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, *,
                apply_et: bool = True, apply_rt: bool = True):
        """
        Convenience forward that returns transformed (h_r, r_ht, t_r).
        Shapes are the same as inputs.
        """
        h_r = head
        t_r = tail
        r_ht = relation
        if apply_et:
            h_r = self.transform_entity(head, relation)
            t_r = self.transform_entity(tail, relation)
        if apply_rt:
            r_ht = self.transform_relation(head, tail, relation)
        return h_r, r_ht, t_r
