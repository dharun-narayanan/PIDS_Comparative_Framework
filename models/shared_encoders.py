"""
Shared encoder architectures for PIDS models.

This module contains common GNN encoder implementations used across
multiple models (MAGIC, Kairos, Orthrus, ThreaTrace, Continuum_FL).

Encoders:
- GraphAttentionNetwork (GAT) - Multi-head attention over graphs
- GraphSAGE - Neighborhood aggregation encoder
- GraphTransformer - Transformer-based graph encoder
- TemporalGraphNetwork (TGN) - Temporal graph encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv, GATConv as PyGGATConv
from typing import Optional, Callable


# ============================================================================
# Utility Functions
# ============================================================================

def create_activation(name: str) -> nn.Module:
    """Create activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'prelu': nn.PReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'identity': nn.Identity(),
    }
    return activations.get(name.lower(), nn.ReLU())


def create_norm(name: str, dim: int) -> Optional[nn.Module]:
    """Create normalization layer by name."""
    if name is None or name.lower() == 'none':
        return None
    elif name.lower() == 'batch':
        return nn.BatchNorm1d(dim)
    elif name.lower() == 'layer':
        return nn.LayerNorm(dim)
    elif name.lower() == 'instance':
        return nn.InstanceNorm1d(dim)
    else:
        return None


# ============================================================================
# Graph Attention Network (GAT) Encoder
# Used by: MAGIC, Continuum_FL, ThreaTrace, Orthrus
# ============================================================================

class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network encoder.
    
    Implements graph attention mechanism with multi-head attention,
    residual connections, and optional normalization.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads per layer
        dropout: Dropout rate
        activation: Activation function name
        residual: Whether to use residual connections
        norm: Normalization type ('batch', 'layer', or None)
        concat_heads: Whether to concatenate multi-head outputs
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = 'relu',
        residual: bool = True,
        norm: Optional[str] = 'batch',
        concat_heads: bool = True
    ):
        super(GATEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        
        # First layer
        self.convs.append(
            PyGGATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=concat_heads
            )
        )
        first_out = hidden_channels * num_heads if concat_heads else hidden_channels
        self.norms.append(create_norm(norm, first_out))
        self.acts.append(create_activation(activation))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                PyGGATConv(
                    first_out,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=concat_heads
                )
            )
            self.norms.append(create_norm(norm, first_out))
            self.acts.append(create_activation(activation))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                PyGGATConv(
                    first_out,
                    out_channels,
                    heads=1,
                    dropout=dropout,
                    concat=False
                )
            )
            self.norms.append(create_norm(norm, out_channels))
            self.acts.append(nn.Identity())
        
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, return_all_layers=False):
        """
        Forward pass through GAT layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes (optional)
            return_all_layers: Return outputs from all layers
            
        Returns:
            Node embeddings [num_nodes, out_channels]
            (optionally: list of embeddings from all layers)
        """
        hidden_list = []
        
        for i, (conv, norm, act) in enumerate(zip(self.convs, self.norms, self.acts)):
            x_in = x
            x = conv(x, edge_index)
            
            if norm is not None:
                x = norm(x)
            
            if act is not None:
                x = act(x)
            
            if self.residual and x_in.shape == x.shape and i > 0:
                x = x + x_in
            
            x = self.dropout(x)
            hidden_list.append(x)
        
        if return_all_layers:
            return x, hidden_list
        return x


# ============================================================================
# GraphSAGE Encoder
# Used by: ThreaTrace, Orthrus
# ============================================================================

class SAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for inductive learning.
    
    Samples and aggregates features from node neighborhoods.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of SAGE layers
        dropout: Dropout rate
        activation: Activation function name
        normalize: Whether to normalize embeddings
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = 'relu',
        normalize: bool = False
    ):
        super(SAGEEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=normalize))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=normalize))
        
        self.activation = create_activation(activation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, return_all_layers=False):
        """
        Forward pass through GraphSAGE layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            return_all_layers: Return outputs from all layers
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        hidden_list = []
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.bns):
                x = self.bns[i](x)
            
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
            
            hidden_list.append(x)
        
        if return_all_layers:
            return x, hidden_list
        return x


# ============================================================================
# Graph Transformer Encoder
# Used by: Kairos, Orthrus
# ============================================================================

class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder using TransformerConv.
    
    Applies self-attention mechanism over graph structure with
    edge features.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        edge_dim: Edge feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        activation: Activation function name
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int = 0,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        super(GraphTransformerEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            TransformerConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim if edge_dim > 0 else None
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None
                )
            )
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                TransformerConv(
                    hidden_channels * num_heads,
                    out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None
                )
            )
        
        self.activation = create_activation(activation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, return_all_layers=False):
        """
        Forward pass through transformer layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            return_all_layers: Return outputs from all layers
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        hidden_list = []
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
            
            hidden_list.append(x)
        
        if return_all_layers:
            return x, hidden_list
        return x


# ============================================================================
# Time Encoder (for Temporal Models)
# Used by: Kairos, TGN-based models
# ============================================================================

class TimeEncoder(nn.Module):
    """
    Time encoding for temporal graphs.
    
    Encodes relative time differences into learnable representations.
    Used in temporal graph networks like Kairos and TGN.
    
    Args:
        time_dim: Dimension of time encoding
    """
    
    def __init__(self, time_dim: int):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
    
    def forward(self, t):
        """
        Encode time values.
        
        Args:
            t: Time tensor [num_edges] or [num_edges, 1]
            
        Returns:
            Time encodings [num_edges, time_dim]
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        # Cosine-based time encoding
        time_encoding = torch.cos(self.w(t))
        return time_encoding


# ============================================================================
# Encoder Factory
# ============================================================================

def get_encoder(encoder_type: str, config: dict) -> nn.Module:
    """
    Factory function to create encoder based on type and configuration.
    
    Args:
        encoder_type: Type of encoder ('gat', 'sage', 'transformer', 'time')
        config: Configuration dictionary with encoder parameters
        
    Returns:
        Encoder module
        
    Example config:
        {
            'in_channels': 128,
            'hidden_channels': 256,
            'out_channels': 128,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1,
            'activation': 'relu'
        }
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type in ['gat', 'graph_attention']:
        return GATEncoder(**config)
    
    elif encoder_type in ['sage', 'graphsage']:
        return SAGEEncoder(**config)
    
    elif encoder_type in ['transformer', 'graph_transformer']:
        return GraphTransformerEncoder(**config)
    
    elif encoder_type == 'time':
        return TimeEncoder(config.get('time_dim', 100))
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================================
# Multi-Encoder Wrapper (for models that use multiple encoders)
# ============================================================================

class MultiEncoder(nn.Module):
    """
    Wrapper for models that use multiple encoder types.
    
    Can combine spatial and temporal encoders, or use different
    encoders for different graph types.
    
    Args:
        encoders: Dictionary mapping encoder names to encoder modules
        aggregation: How to aggregate outputs ('concat', 'sum', 'mean')
    """
    
    def __init__(self, encoders: dict, aggregation: str = 'concat'):
        super(MultiEncoder, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.aggregation = aggregation
    
    def forward(self, x, edge_index, edge_attr=None, encoder_name=None, **kwargs):
        """
        Forward pass through one or all encoders.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge attributes
            encoder_name: Specific encoder to use (None = use all)
            
        Returns:
            Combined embeddings from encoder(s)
        """
        if encoder_name is not None:
            return self.encoders[encoder_name](x, edge_index, edge_attr, **kwargs)
        
        # Use all encoders
        outputs = []
        for encoder in self.encoders.values():
            out = encoder(x, edge_index, edge_attr, **kwargs)
            outputs.append(out)
        
        # Aggregate
        if self.aggregation == 'concat':
            return torch.cat(outputs, dim=-1)
        elif self.aggregation == 'sum':
            return torch.stack(outputs).sum(dim=0)
        elif self.aggregation == 'mean':
            return torch.stack(outputs).mean(dim=0)
        else:
            return outputs[0]
