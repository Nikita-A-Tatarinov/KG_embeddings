# models/__init__.py
from .complex_ import ComplEx as ComplEx
from .distmult import DistMult as DistMult
from .pairre import PairRE as PairRE
from .registry import (
    create_model as create_model,
)
from .registry import (
    get_model_class as get_model_class,
)
from .registry import (
    list_models as list_models,
)
from .registry import (
    register_model as register_model,
)
from .rotate import RotatE as RotatE
from .rotatev2 import RotatEv2 as RotatEv2

# Import modules to trigger @register_model side-effects and optionally re-export classes
from .transe import TransE as TransE
from .transh import TransH as TransH

__all__ = [
    # registry API
    "create_model",
    "get_model_class",
    "list_models",
    "register_model",
    # model classes (optional; handy for type hints / direct imports)
    "TransE",
    "DistMult",
    "ComplEx",
    "RotatE",
    "RotatEv2",
    "PairRE",
    "TransH",
]
