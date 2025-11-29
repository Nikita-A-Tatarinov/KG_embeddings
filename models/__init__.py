# models/__init__.py
from .complex_ import ComplEx as ComplEx
from .context_pooling import ContextPooling
from .distmult import DistMult as DistMult

# models/__init__.py
from .kg_model import KGModel
from .mi import MI_Module
from .pairre import PairRE as PairRE
from .registry import (
    create_model as create_model,
)
from .registry import list_models, register_model
from .rotate import RotatE
from .rotatev2 import RotatEv2

# Wrappers
from .rscf import RSCFModule
from .transe import TransE
from .transh import TransH

# Import modules to trigger @register_model side-effects and optionally re-export classes

__all__ = [
    # registry API
    "KGModel",
    "create_model",
    "get_model_class",
    "list_models",
    "register_model",
    "TransE",
    "DistMult",
    "ComplEx",
    "RotatE",
    "RotatEv2",
    "PairRE",
    "TransH",
    "RSCFModule",
    "MI_Module",
    "ContextPooling",
]
