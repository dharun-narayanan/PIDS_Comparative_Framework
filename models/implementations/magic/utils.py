"""
Utility functions for MAGIC model implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import random


def create_activation(name):
    """Create activation function by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Activation {name} not implemented")


def create_norm(name):
    """Create normalization layer by name."""
    if name == "batchnorm" or name == "BatchNorm":
        return nn.BatchNorm1d
    elif name == "layernorm" or name == "LayerNorm":
        return nn.LayerNorm
    elif name is None:
        return None
    else:
        raise NotImplementedError(f"Normalization {name} not implemented")


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
