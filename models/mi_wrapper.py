from __future__ import annotations

import torch
import types
from typing import Optional

from .mi import MI_Module


def attach_mi(model, q_dim: Optional[int] = None, use_info_nce: bool = True, use_jsd: bool = False):
    """
    Attach an MI_Module to a KGModel instance.

    This registers `model._mi` and adds a `model.compute_mi_loss(sample, neg_size=... )` method
    that returns a scalar MI loss combining InfoNCE and JSD as configured.
    """
    # infer dims
    e_dim = model.entity_embedding.size(1)
    r_dim = model.relation_embedding.size(1)
    q_dim = q_dim or (e_dim + r_dim)
    # q input dim is concat(h,r) => e_dim + r_dim; t input dim is e_dim
    mi = MI_Module(q_in_dim=(e_dim + r_dim), t_in_dim=e_dim)
    model._mi = mi
    model.add_module("_mi", mi)

    def _phi(self, head_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        # simple composition: concat (h, r) and linearize via a tiny linear layer stored on model
        # create projector lazily
        if not hasattr(self, "_mi_phi"):
            self._mi_phi = torch.nn.Linear(head_emb.size(-1) + rel_emb.size(-1), q_dim).to(head_emb.device)
        x = torch.cat([head_emb, rel_emb], dim=-1)
        return self._mi_phi(x)

    def compute_mi_loss(self, sample: torch.LongTensor, neg_size: int = 32, *, device=None):
        """
        sample: (B,3) tensor of (h,r,t)
        neg_size: number of negatives per positive for InfoNCE
        returns scalar loss
        """
        device = device or self.entity_embedding.device
        B = sample.size(0)
        h_idx = sample[:, 0]
        r_idx = sample[:, 1]
        t_idx = sample[:, 2]

        h_emb = torch.index_select(self.entity_embedding, 0, h_idx).to(device)
        r_emb = torch.index_select(self.relation_embedding, 0, r_idx).to(device)
        t_emb = torch.index_select(self.entity_embedding, 0, t_idx).to(device)

        q = _phi(self, h_emb, r_emb)  # (B, q_dim)

        total_loss = 0.0
        count = 0

        if use_info_nce:
            # sample negative tails uniformly
            neg_idx = torch.randint(0, self.entity_embedding.size(0), (B, neg_size), device=device)
            neg_t = torch.index_select(self.entity_embedding, 0, neg_idx.view(-1)).view(B, neg_size, -1)
            loss_info = self._mi.info_nce_loss(q, t_emb, neg_t)
            total_loss = total_loss + loss_info
            count += 1

        if use_jsd:
            # build negative queries from other batch elements (roll)
            neg_q_idx = torch.roll(h_idx, shifts=1)
            neg_h_emb = torch.index_select(self.entity_embedding, 0, neg_q_idx).to(device)
            neg_q = _phi(self, neg_h_emb, r_emb).unsqueeze(1)  # (B,1,q_dim)
            # expand to a small M by repeating
            neg_q = neg_q.expand(B, min(4, B), -1)
            loss_jsd = self._mi.jsd_loss(q, t_emb, neg_q)
            total_loss = total_loss + loss_jsd
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / float(count)

    model.compute_mi_loss = types.MethodType(compute_mi_loss, model)
    return mi
