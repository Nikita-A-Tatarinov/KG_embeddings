# models/__init__.py
from .complex_ import ComplEx as ComplEx
from .distmult import DistMult as DistMult
from .pairre import PairRE as PairRE
from .registry import (
    create_model as create_model,
)
from .registry import create_model, list_models, register_model

# models/__init__.py
from .kg_model import KGModel
from .transe import TransE
from .distmult import DistMult
from .complex_ import ComplEx
from .rotate import RotatE
from .rotatev2 import RotatEv2
from .pairre import PairRE
from .transh import TransH
from .context_pooling import ContextPooling

# Wrappers
from .rscf import RSCFModule
from .mi import MI_Module

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
    "TransE",
    "DistMult",
    "ComplEx",
    "RotatE",
    "RotatEv2",
    "PairRE",
    "TransH",
    "RSCFModule",
    "MI_Module",
    "ContextPooling"
]
