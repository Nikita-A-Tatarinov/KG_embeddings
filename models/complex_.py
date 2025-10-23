# models/complex_.py
import torch

from .kg_model import KGModel
from .registry import register_model


@register_model("ComplEx", "complex", "complexe")
class ComplEx(KGModel):
    ENTITY_FACTOR = 2  # re|im
    RELATION_FACTOR = 2  # re|im

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_r, im_r = torch.chunk(relation, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_s = re_r * re_t + im_r * im_t
            im_s = re_r * im_t - im_r * re_t
            x = re_h * re_s + im_h * im_s
        else:
            re_s = re_h * re_r - im_h * im_r
            im_s = re_h * im_r + im_h * re_r
            x = re_s * re_t + im_s * im_t

        return x.sum(dim=2)
