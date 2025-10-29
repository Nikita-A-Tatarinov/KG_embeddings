# models/transe.py
import torch

from .kg_model import KGModel
from .registry import register_model


@register_model("TransE", "transe")
class TransE(KGModel):
    ENTITY_FACTOR = 1
    RELATION_FACTOR = 1

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        if mode == "head-batch":
            diff = head + (relation - tail)
        else:  # 'single' or 'tail-batch'
            diff = (head + relation) - tail
        # L1 distance as logit via margin trick
        return self.gamma - torch.norm(diff, p=1, dim=2)
