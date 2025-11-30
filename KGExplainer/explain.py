# a small utility to get an explanation subgraph + fidelity for a given triple

from __future__ import annotations

from typing import Dict

import dgl
import torch

from .graph import k_hop_enclosing_subgraph
from .evaluator import HeteroGAT


@torch.no_grad()
def explain_triple(
    h: int,
    r: int,
    t: int,
    g_full: dgl.DGLHeteroGraph,
    teacher,
    evaluator: HeteroGAT,
    device,
    k_hop: int = 2,
) -> Dict[str, object]:
    """
    Return:
      - subgraph g_expl
      - teacher_score_full
      - evaluator_score_subgraph
      - fidelity (ratio)
    """
    triple = torch.tensor([h, r, t], dtype=torch.long, device=device)

    # teacher score using full model
    teacher.eval()
    logits_full = teacher(triple.view(1, 3), mode="single").view(-1)[0]
    teacher_score_full = logits_full.item()

    # explanation subgraph: k-hop enclosing
    g_expl = k_hop_enclosing_subgraph(g_full, h, t, k_hop=k_hop)
    g_expl = dgl.batch([g_expl]).to(device)

    evaluator.eval()
    _, score_sub = evaluator(g_expl)
    score_sub = score_sub.view(-1)[0].item()

    fidelity = score_sub / teacher_score_full if teacher_score_full != 0 else 0.0

    return {
        "subgraph": g_expl,
        "teacher_score_full": teacher_score_full,
        "evaluator_score_subgraph": score_sub,
        "fidelity": fidelity,
    }
