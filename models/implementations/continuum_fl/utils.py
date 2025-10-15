"""
Utility Functions for Continuum_FL
"""

import torch
import torch.nn as nn
from functools import partial
import numpy as np
import random
import torch.optim as optim


def create_optimizer(opt, model, lr, weight_decay):
    """
    Create optimizer based on configuration
    
    Args:
        opt: Optimizer name
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        optimizer: Configured optimizer
    """
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    optimizer = None
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer: {opt}")
    return optimizer


def random_shuffle(x, y):
    """Randomly shuffle two arrays in sync"""
    idx = list(range(len(x)))
    random.shuffle(idx)
    return x[idx], y[idx]


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def create_activation(name):
    """
    Create activation function by name
    
    Args:
        name: Activation function name
        
    Returns:
        activation: Activation module
    """
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    """
    Create normalization layer by name
    
    Args:
        name: Normalization type
        
    Returns:
        norm: Normalization layer class
    """
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class NormLayer(nn.Module):
    """Custom normalization layer supporting batch, layer, and graph norms"""
    
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        """Apply normalization"""
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class Pooling(nn.Module):
    """Graph pooling layer for aggregating node features"""
    
    def __init__(self, pooler):
        """
        Args:
            pooler: Pooling type ('mean', 'sum', or 'max')
        """
        super(Pooling, self).__init__()
        self.pooler = pooler

    def forward(self, graph, feat, t=None):
        """
        Pool node features
        
        Args:
            graph: DGL graph
            feat: Node features
            t: Optional node type(s) to filter
            
        Returns:
            pooled: Pooled features
        """
        feat = feat
        with graph.local_scope():
            if t is None:
                if self.pooler == 'mean':
                    return feat.mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat.sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat.max(0, keepdim=True)[0]
                else:
                    raise NotImplementedError
            elif isinstance(t, int):
                mask = (graph.ndata['type'] == t)
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)[0]
                else:
                    raise NotImplementedError
            else:
                mask = (graph.ndata['type'] == t[0])
                for i in range(1, len(t)):
                    mask |= (graph.ndata['type'] == t[i])
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)[0]
                else:
                    raise NotImplementedError


def setup_continuum_fl_model(config):
    """
    Factory function to create Continuum_FL model
    
    Args:
        config: Model configuration dictionary with keys:
            - n_dim: Node feature dimension
            - e_dim: Edge feature dimension
            - hidden_dim: Hidden dimension
            - out_dim: Output dimension
            - n_layers: Number of GAT layers
            - n_heads: Number of attention heads
            - device: Device to place model
            - number_snapshot: Number of temporal snapshots
            - activation: Activation function name
            - feat_drop: Feature dropout rate
            - negative_slope: LeakyReLU slope
            - residual: Use residual connections
            - norm: Normalization type
            - pooling: Pooling type
            - loss_fn: Loss function name
            - alpha_l: Loss alpha parameter
            - use_all_hidden: Use all hidden layers
            
    Returns:
        model: STGNN_AutoEncoder model
    """
    from .model import STGNN_AutoEncoder
    
    model = STGNN_AutoEncoder(
        n_dim=config.get('n_dim', 64),
        e_dim=config.get('e_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        out_dim=config.get('out_dim', 128),
        n_layers=config.get('n_layers', 2),
        n_heads=config.get('n_heads', 4),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        number_snapshot=config.get('number_snapshot', 5),
        activation=config.get('activation', 'relu'),
        feat_drop=config.get('feat_drop', 0.1),
        negative_slope=config.get('negative_slope', 0.2),
        residual=config.get('residual', True),
        norm=config.get('norm', 'layernorm'),
        pooling=config.get('pooling', 'mean'),
        loss_fn=config.get('loss_fn', 'sce'),
        alpha_l=config.get('alpha_l', 2),
        use_all_hidden=config.get('use_all_hidden', True)
    )
    
    return model
