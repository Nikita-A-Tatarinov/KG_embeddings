# train/config.py
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(x) for x in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            data: dict[str, Any] = json.load(f)
        else:
            data = yaml.safe_load(f)
    return _to_ns(data)


def ensure_out_dir(cfg: SimpleNamespace) -> str:
    out = getattr(cfg.train, "out_dir", "./runs/default")
    os.makedirs(out, exist_ok=True)
    return out
