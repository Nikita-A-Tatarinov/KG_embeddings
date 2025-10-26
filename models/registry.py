# models/registry.py
from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def _canon(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


def register_model(*names):
    """
    Decorator to register a model class under one or more names.
    Usage:
        @register_model("TransE", "transe")
        class TransE(KGModel): ...
    """

    def deco(cls):
        for n in names:
            key = _canon(n)
            if key in _REGISTRY and _REGISTRY[key] is not cls:
                raise ValueError(f"Model name '{n}' already registered by {_REGISTRY[key].__name__}")
            _REGISTRY[key] = cls
        return cls

    return deco


def get_model_class(name: str):
    key = _canon(name)
    if key not in _REGISTRY:
        known = ", ".join(sorted({k for k in _REGISTRY}))
        raise KeyError(f"Unknown model '{name}'. Known: {known}")
    return _REGISTRY[key]


def create_model(name: str, **kwargs):
    cls = get_model_class(name)
    return cls(**kwargs)


def list_models():
    # Return deduped pretty names
    seen = set()
    out = []
    for k, v in _REGISTRY.items():
        nm = v.__name__
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return sorted(out)
