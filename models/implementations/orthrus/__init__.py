"""
Orthrus Model Implementation
Standalone implementation for PIDS Comparative Framework.

Paper: "Orthrus: High Quality Attribution in Provenance-based IDS"
USENIX Security 2025
"""

from .model import Orthrus, OrthrusEncoder
from .encoders import GraphTransformer, get_encoder
from .decoders import get_decoders
from .utils import setup_orthrus_model

__all__ = [
    'Orthrus',
    'OrthrusEncoder',
    'GraphTransformer',
    'get_encoder',
    'get_decoders',
    'setup_orthrus_model'
]
