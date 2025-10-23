# models/pairre.py
import torch
import torch.nn.functional as F

from .kg_model import KGModel
from .registry import register_model


@register_model("PairRE", "pairre")
class PairRE(KGModel):
    ENTITY_FACTOR = 1
    RELATION_FACTOR = 2  # r1 | r2

    def __init__(self, *args, nonneg: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nonneg = bool(nonneg)

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        r1, r2 = torch.chunk(relation, 2, dim=2)
        if self.nonneg:
            r1 = F.softplus(r1)
            r2 = F.softplus(r2)

        head = F.normalize(head, p=2, dim=-1)
        tail = F.normalize(tail, p=2, dim=-1)

        diff = head * r1 - tail * r2
        return self.gamma - torch.norm(diff, p=1, dim=2)
