# train/checkpoint.py
from __future__ import annotations

import os
from typing import Any

import torch


def save_checkpoint(
    out_dir: str,
    tag: str,
    trainer_state: dict[str, Any],
    model_obj,
    optimizer,
    scheduler=None,
    model_config: dict[str, Any] = None,
):
    """Save checkpoint with optional model configuration.

    Args:
        model_config: Dict with keys like 'model_name', 'base_dim', 'nentity', 
                      'nrelation', 'gamma', 'med_enabled', 'd_list', etc.
                      This enables proper model reconstruction during evaluation.
    """
    path = os.path.join(out_dir, f"ckpt_{tag}.pt")
    payload = {
        "trainer": trainer_state,
        "model": model_obj.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    # Add model configuration if provided
    if model_config is not None:
        payload["model_config"] = model_config

    torch.save(payload, path)
    return path


def load_checkpoint(path: str, model_obj, optimizer=None, scheduler=None):
    data = torch.load(path, map_location="cpu")
    model_obj.load_state_dict(data["model"], strict=False)
    if optimizer is not None and data.get("optimizer") is not None:
        optimizer.load_state_dict(data["optimizer"])
    if scheduler is not None and data.get("scheduler") is not None:
        scheduler.load_state_dict(data["scheduler"])
    return data.get("trainer", {})
