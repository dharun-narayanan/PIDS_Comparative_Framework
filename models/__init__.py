"""
Model registry and initialization.

This module provides the ModelBuilder for dynamic model construction
from YAML configurations and exports all shared components.
"""

from models.model_builder import ModelBuilder, GenericModel
from models.shared_encoders import (
    GATEncoder,
    SAGEEncoder,
    GraphTransformerEncoder,
    TimeEncoder,
    MultiEncoder
)
from models.shared_decoders import (
    EdgeDecoder,
    NodeDecoder,
    ContrastiveDecoder,
    ReconstructionDecoder,
    AnomalyDecoder,
    InnerProductDecoder
)

__all__ = [
    'ModelBuilder',
    'GenericModel',
    # Encoders
    'GATEncoder',
    'SAGEEncoder',
    'GraphTransformerEncoder',
    'TimeEncoder',
    'MultiEncoder',
    # Decoders
    'EdgeDecoder',
    'NodeDecoder',
    'ContrastiveDecoder',
    'ReconstructionDecoder',
    'AnomalyDecoder',
    'InnerProductDecoder',
]

# Utility function for listing available models
def list_available_models():
    """List all available models from configs/models/."""
    from pathlib import Path
    config_dir = Path(__file__).parent.parent / "configs" / "models"
    model_configs = list(config_dir.glob("*.yaml"))
    return [f.stem for f in model_configs if f.stem != "template"]

# Model configuration templates
MODEL_CONFIGS = {
    'magic': {
        'num_hidden': 256,
        'num_layers': 4,
        'mask_rate': 0.3,
        'n_dim': 128,
        'e_dim': 64,
    },
    'magic_streamspot': {
        'num_hidden': 256,
        'num_layers': 4,
        'mask_rate': 0.3,
    },
    'magic_darpa': {
        'num_hidden': 64,
        'num_layers': 3,
        'mask_rate': 0.5,
    },
    'kairos': {
        'in_channels': 100,
        'out_channels': 100,
        'msg_dim': 100,
        'memory_dim': 100,
    },
    'orthrus': {
        'num_nodes': 100000,
        'in_dim': 100,
        'out_dim': 100,
        'use_contrastive_learning': True,
    },
    'threatrace': {
        'hidden_dim': 128,
        'num_layers': 2,
    },
}

def get_model_config(model_name: str):
    """Get default configuration for a model."""
    return MODEL_CONFIGS.get(model_name, {})
