# models/transh.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kg_model import KGModel
from .registry import register_model


@register_model("TransH", "transh")
class TransH(KGModel):
    ENTITY_FACTOR = 1
    RELATION_FACTOR = 1

    def _post_alloc_extras(self):
        # One normal per relation, base-dim (croppable)
        self.norm_vector = nn.Parameter(
            torch.zeros(
                self.nrelation,
                self.max_base_dim,
                device=self.relation_embedding.device,
                dtype=self.relation_embedding.dtype,
            )
        )

    def _post_init_extras(self):
        nn.init.uniform_(self.norm_vector, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def _gather_norm(self, rel_ids: torch.Tensor, d: int):
        n = torch.index_select(self.norm_vector, 0, rel_ids).unsqueeze(1)  # (B,1,dmax)
        return n[..., :d]

    def _project(self, e: torch.Tensor, n: torch.Tensor):
        n = F.normalize(n, p=2, dim=-1)
        return e - (e * n).sum(dim=-1, keepdim=True) * n

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        d = crop_dim if crop_dim is not None else self.max_base_dim
        r_norm = self._gather_norm(rel_ids, d)

        h = self._project(head, r_norm)
        t = self._project(tail, r_norm)

        if mode == "head-batch":
            diff = h + (relation - t)
        else:
            diff = (h + relation) - t

        return self.gamma - torch.norm(diff, p=1, dim=2)
