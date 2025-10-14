"""
Model registry and initialization.

This module imports all PIDS model wrappers and registers them
with the ModelRegistry for easy access.
"""

from models.base_model import BasePIDSModel, ModelRegistry

# Import all model wrappers
# These imports trigger the @ModelRegistry.register decorators
try:
    from models.magic_wrapper import MAGICModel, MAGICStreamSpot, MAGICDARPA
except ImportError as e:
    print(f"Warning: Could not import MAGIC model: {e}")

try:
    from models.kairos_wrapper import KairosModel
except ImportError as e:
    print(f"Warning: Could not import Kairos model: {e}")

try:
    from models.orthrus_wrapper import OrthrusModel
except ImportError as e:
    print(f"Warning: Could not import Orthrus model: {e}")

try:
    from models.threatrace_wrapper import ThreaTraceModel
except ImportError as e:
    print(f"Warning: Could not import ThreaTrace model: {e}")

try:
    from models.continuum_fl_wrapper import ContinuumFLModel, ContinuumFLStreamSpot, ContinuumFLDARPA
except ImportError as e:
    print(f"Warning: Could not import Continuum_FL model: {e}")

__all__ = [
    'BasePIDSModel',
    'ModelRegistry',
    'MAGICModel',
    'KairosModel',
    'OrthrusModel',
    'ThreaTraceModel',
    'ContinuumFLModel',
]

# Print registered models
def list_available_models():
    """List all registered models."""
    return ModelRegistry.list_models()

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
