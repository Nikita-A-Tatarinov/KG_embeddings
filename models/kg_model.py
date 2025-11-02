# models/kg_model.py
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGModel(nn.Module):
    """
    Croppable, model-adaptive KGE base.

    Subclasses define class vars:
      - ENTITY_FACTOR: 1 for R^d, 2 for (real|imag) entities
      - RELATION_FACTOR: 1 for d, 2 for (r1|r2) or (real|imag) relations
      - REQUIRE_EVEN_D: set True if crop_dim must be even (e.g., RotatEv2)

    Each subclass must implement:
      - def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None) -> torch.Tensor

    Public API:
      - forward(sample, mode='single', crop_dim=None) -> logits
      - scores_for_dims(sample, dims, mode='single') -> {d: logits}
    """

    ENTITY_FACTOR: int = 1
    RELATION_FACTOR: int = 1
    REQUIRE_EVEN_D: bool = False

    def __init__(
        self,
        nentity: int,
        nrelation: int,
        base_dim: int,
        gamma: float,
        evaluator=None,
        epsilon: float = 2.0,
        init: str = "uniform",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.max_base_dim = int(base_dim)
        self.evaluator = evaluator

        # Hyper-parameters as tensors to avoid CPU-GPU syncs via .item()
        self.gamma = nn.Parameter(torch.tensor(float(gamma), device=device, dtype=dtype), requires_grad=False)
        self.epsilon = float(epsilon)
        emb_range = (float(gamma) + self.epsilon) / float(self.max_base_dim)
        self.embedding_range = nn.Parameter(torch.tensor(emb_range, device=device, dtype=dtype), requires_grad=False)

        # Derived embedding sizes
        e_dim = self.ENTITY_FACTOR * self.max_base_dim
        r_dim = self.RELATION_FACTOR * self.max_base_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, e_dim, device=device, dtype=dtype))
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, r_dim, device=device, dtype=dtype))

        # Optional extras (e.g., TransH will add self.norm_vector)
        self._post_alloc_extras()

        # Init
        if init == "uniform":
            nn.init.uniform_(self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            nn.init.uniform_(self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            self._post_init_extras()
        else:
            raise ValueError(f"Unknown init '{init}'")

    # ---- subclass hooks -------------------------------------------------------
    def _post_alloc_extras(self):
        """Subclasses can allocate extra parameters (e.g., TransH normals)."""
        pass

    def _post_init_extras(self):
        """Subclasses can initialize extra parameters."""
        pass

    # ---- common indexing / cropping helpers ----------------------------------
    def _index_full(self, sample, mode: str):
        """
        Return (head, relation, tail, rel_ids) at full dims.
        Shapes:
          - 'single'     : (B,1,D_e), (B,1,D_r), (B,1,D_e)
          - 'head-batch' : (B,K,D_e), (B,1,D_r), (B,1,D_e)   (negatives in head)
          - 'tail-batch' : (B,1,D_e), (B,1,D_r), (B,K,D_e)   (negatives in tail)
        rel_ids: (B,)
        """
        if mode == "single":
            head = torch.index_select(self.entity_embedding, 0, sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, 0, sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, 0, sample[:, 2]).unsqueeze(1)
            rel_ids = sample[:, 1]

        elif mode == "head-batch":
            tail_part, head_part = sample
            B, K = head_part.size(0), head_part.size(1)
            head = torch.index_select(self.entity_embedding, 0, head_part.view(-1)).view(B, K, -1)
            relation = torch.index_select(self.relation_embedding, 0, tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, 0, tail_part[:, 2]).unsqueeze(1)
            rel_ids = tail_part[:, 1]

        elif mode == "tail-batch":
            head_part, tail_part = sample
            B, K = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(self.entity_embedding, 0, head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, 0, head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, 0, tail_part.view(-1)).view(B, K, -1)
            rel_ids = head_part[:, 1]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return head, relation, tail, rel_ids

    def _slice_entity(self, E: torch.Tensor, d: int) -> torch.Tensor:
        need = self.ENTITY_FACTOR * d
        return E[..., :need]

    def _slice_relation(self, R: torch.Tensor, d: int) -> torch.Tensor:
        need = self.RELATION_FACTOR * d
        return R[..., :need]

    def _check_dim_ok(self, d: int):
        if not (1 <= int(d) <= self.max_base_dim):
            raise ValueError(f"crop_dim {d} must be in [1, {self.max_base_dim}]")
        if self.REQUIRE_EVEN_D and (d % 2 != 0):
            raise ValueError(f"{self.__class__.__name__} requires even crop_dim; got {d}")

    # ---- public API -----------------------------------------------------------
    def forward(self, sample, mode: str = "single", crop_dim: Optional[int] = None) -> torch.Tensor:
        """Compute logits at full dim (crop_dim=None) or at a sub-dimension crop_dim."""
        head, relation, tail, rel_ids = self._index_full(sample, mode)
        if crop_dim is not None:
            self._check_dim_ok(crop_dim)
            head = self._slice_entity(head, crop_dim)
            tail = self._slice_entity(tail, crop_dim)
            relation = self._slice_relation(relation, crop_dim)
        return self.score(head, relation, tail, mode, rel_ids=rel_ids, crop_dim=crop_dim)

    @torch.no_grad()
    def scores_for_dims(self, sample, dims: Iterable[int], mode: str = "single") -> dict[int, torch.Tensor]:
        """Return logits dict for multiple crop_dims in one pass (looped)."""
        out = {}
        # index once
        head, relation, tail, rel_ids = self._index_full(sample, mode)
        for d in dims:
            self._check_dim_ok(int(d))
            h = self._slice_entity(head, int(d))
            r = self._slice_relation(relation, int(d))
            t = self._slice_entity(tail, int(d))
            out[int(d)] = self.score(h, r, t, mode, rel_ids=rel_ids, crop_dim=int(d))
        return out

    # ---- to be overridden -----------------------------------------------------
    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None) -> torch.Tensor:
        raise NotImplementedError

    # ---- (Optional) OpenKE-style helpers for compatibility -------------------
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        Generic OpenKE-style training step.
        Expects model.forward(...) to output logits (higher is better).
        """
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if getattr(args, "cuda", False):
            positive_sample = positive_sample.cuda(non_blocking=True)
            negative_sample = negative_sample.cuda(non_blocking=True)
            subsampling_weight = subsampling_weight.cuda(non_blocking=True)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if getattr(args, "negative_adversarial_sampling", False):
            temp = getattr(args, "adversarial_temperature", 1.0)
            neg_log = (F.softmax(negative_score * temp, dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)
            negative_score = neg_log
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(1)

        if getattr(args, "uni_weight", False):
            pos_loss = -positive_score.mean()
            neg_loss = -negative_score.mean()
        else:
            w = subsampling_weight
            pos_loss = -(w * positive_score).sum() / w.sum()
            neg_loss = -(w * negative_score).sum() / w.sum()

        loss = (pos_loss + neg_loss) / 2

        reg = getattr(args, "regularization", 0.0)
        if reg:
            regularization = reg * (model.entity_embedding.norm(p=3) ** 3 + model.relation_embedding.norm(p=3) ** 3)
            loss = loss + regularization

        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.detach()),
            "pos_loss": float(pos_loss.detach()),
            "neg_loss": float(neg_loss.detach()),
        }
