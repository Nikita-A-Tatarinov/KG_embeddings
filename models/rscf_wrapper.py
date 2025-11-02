from __future__ import annotations

import types

from .rscf import RSCFModule


def attach_rscf(model, use_et: bool = True, use_rt: bool = True) -> RSCFModule:
    """
    Attaches an RSCFModule to a KGModel instance in-place.

    After attaching, model._rscf will hold the module (registered as a submodule), and
    model.score will be wrapped so that the RSCF transformations are applied before
    delegating to the original scoring implementation.

    Returns the created RSCFModule.
    """
    # infer dims from model
    e_dim = model.entity_embedding.size(1)
    r_dim = model.relation_embedding.size(1)
    rscf = RSCFModule(e_dim=e_dim, r_dim=r_dim)
    # register as attribute/module so its parameters are included in model.parameters()
    model._rscf = rscf
    model.add_module("_rscf", rscf)

    # keep original score
    if getattr(model, "_score_orig", None) is None:
        model._score_orig = model.score

    def _score_with_rscf(self, head, relation, tail, mode, *, rel_ids=None, crop_dim=None):
        # head/relation/tail shapes: (..., D)
        # apply RSCF transforms (note: RSCF expects same trailing dim sizes)
        h_r, r_ht, t_r = model._rscf(head, relation, tail, apply_et=use_et, apply_rt=use_rt)
        # delegate to original score; for TDM models that expect only head transformed (e.g., ComplEX)
        # the wrapped model._score_orig should accept the same shapes and semantics.
        return model._score_orig(h_r, r_ht, t_r, mode, rel_ids=rel_ids, crop_dim=crop_dim)

    # replace
    model.score = types.MethodType(_score_with_rscf, model)
    return rscf
