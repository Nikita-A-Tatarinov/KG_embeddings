# models/distmult.py
from .kg_model import KGModel
from .registry import register_model


@register_model("DistMult", "distmult")
class DistMult(KGModel):
    ENTITY_FACTOR = 1
    RELATION_FACTOR = 1

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        if mode == "head-batch":
            x = head * (relation * tail)
        else:
            x = (head * relation) * tail
        return x.sum(dim=2)
