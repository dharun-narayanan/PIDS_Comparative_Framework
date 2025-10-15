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

def prepare_orthrus_batch(batch):
    """
    Prepare batch data for Orthrus model.
    
    Args:
        batch: Input batch from dataloader
        
    Returns:
        Prepared batch compatible with Orthrus model
    """
    # Basic batch preparation for Orthrus
    return batch

__all__ = [
    'Orthrus',
    'OrthrusEncoder',
    'GraphTransformer',
    'get_encoder',
    'get_decoders',
    'setup_orthrus_model',
    'prepare_orthrus_batch'
]
