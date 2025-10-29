# models/rotate.py
import torch

from .kg_model import KGModel
from .registry import register_model


@register_model("RotatE", "rotate")
class RotatE(KGModel):
    ENTITY_FACTOR = 2  # re|im
    RELATION_FACTOR = 1  # phase

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        pi = torch.tensor(torch.pi, device=head.device, dtype=head.dtype)

        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)

        # relation stores phases; scale to [-pi, pi]
        phase = relation / (self.embedding_range / pi)
        re_r = torch.cos(phase)
        im_r = torch.sin(phase)

        if mode == "head-batch":
            re_s = re_r * re_t + im_r * im_t
            im_s = re_r * im_t - im_r * re_t
            re_s = re_s - re_h
            im_s = im_s - im_h
        else:
            re_s = re_h * re_r - im_h * im_r
            im_s = re_h * im_r + im_h * re_r
            re_s = re_s - re_t
            im_s = im_s - im_t

        # Euclidean across complex dims, then sum across embedding dim
        z = torch.stack([re_s, im_s], dim=0).norm(dim=0)
        return self.gamma - z.sum(dim=2)
