# models/rotatev2.py
import torch

from .kg_model import KGModel
from .registry import register_model


@register_model("RotatEv2", "rotatev2")
class RotatEv2(KGModel):
    ENTITY_FACTOR = 2  # re|im
    RELATION_FACTOR = 2  # phase split into two halves (head/tail)
    REQUIRE_EVEN_D = True  # we split along embedding dim

    def score(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        pi = torch.tensor(torch.pi, device=head.device, dtype=head.dtype)

        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)

        phase = relation / (self.embedding_range / pi)
        # cos/sin, then split for head/tail halves
        re_r = torch.cos(phase)
        im_r = torch.sin(phase)
        re_r_h, re_r_t = torch.chunk(re_r, 2, dim=2)
        im_r_h, im_r_t = torch.chunk(im_r, 2, dim=2)

        re_rot_h = re_h * re_r_h - im_h * im_r_h
        im_rot_h = re_h * im_r_h + im_h * re_r_h

        re_rot_t = re_t * re_r_t - im_t * im_r_t
        im_rot_t = re_t * im_r_t + im_t * re_r_t

        re_s = re_rot_h - re_rot_t
        im_s = im_rot_h - im_rot_t

        z = torch.stack([re_s, im_s], dim=0).norm(dim=0)
        return self.gamma - z.sum(dim=2)
