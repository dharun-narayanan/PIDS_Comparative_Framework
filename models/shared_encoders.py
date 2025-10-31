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
    
    def forward(self, x, edge_index, edge_attr=None, return_all_layers=False):
        """
        Forward pass through GraphSAGE layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes (optional, ignored by SAGE)
            return_all_layers: Return outputs from all layers
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Note: edge_attr is accepted for compatibility but not used by GraphSAGE
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
        encoder_type: Type of encoder ('gat', 'sage', 'transformer', 'time', 'gin', 'glstm', 'linear', 'sum_aggregation')
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
    
    elif encoder_type in ['gin', 'graph_isomorphism']:
        return GINEncoder(**config)
    
    elif encoder_type in ['glstm', 'graph_lstm']:
        return GLSTMEncoder(**config)
    
    elif encoder_type == 'linear':
        return LinearEncoder(**config)
    
    elif encoder_type in ['sum_aggregation', 'sum_pool']:
        return SumAggregationEncoder(**config)
    
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


# ============================================================================
# Graph Isomorphism Network (GIN) Encoder
# Used by: Flash, Velox, other graph classification tasks
# ============================================================================

class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder.
    
    Implements the powerful GIN architecture with edge features support (GINE).
    GIN is theoretically proven to be as powerful as the WL test for graph
    isomorphism, making it excellent for graph classification tasks.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of GIN layers
        dropout: Dropout rate
        activation: Activation function name
        edge_dim: Edge feature dimension (None for GIN, int for GINE)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = 'relu',
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        from torch_geometric.nn import GINConv, GINEConv
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation(activation)
        
        self.convs = nn.ModuleList()
        current_dim = in_channels
        
        for i in range(num_layers):
            # MLP for each GIN layer
            nn_seq = nn.Sequential(
                nn.Linear(current_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            
            if edge_dim is None:
                conv = GINConv(nn_seq)
            else:
                conv = GINEConv(nn_seq, edge_dim=edge_dim)
            
            self.convs.append(conv)
            current_dim = hidden_channels
        
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# Graph LSTM (GLSTM) Encoder  
# Used by: Hierarchical graph processing, temporal dependencies
# ============================================================================

class GLSTMEncoder(nn.Module):
    """
    Graph Long Short-Term Memory (GLSTM) encoder.
    
    Processes graphs with tree-like structures using LSTM-style gating.
    Particularly effective for provenance graphs with hierarchical relationships.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output embedding dimension
        cell_clip: Optional cell state clipping value
        typed_hidden_rep: Include edge type embedding in hidden states
        edge_dim: Edge feature dimension
        num_edge_types: Number of edge types (default: 10)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cell_clip: Optional[float] = None,
        typed_hidden_rep: bool = False,
        edge_dim: Optional[int] = None,
        num_edge_types: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cell_clip = cell_clip
        self.typed_hidden_rep = typed_hidden_rep
        self.edge_dim = edge_dim if edge_dim is not None else in_channels
        self.num_edge_types = num_edge_types
        
        # Input, output, and update gates
        self.W_iou = nn.Linear(in_channels, 3 * out_channels)
        if typed_hidden_rep:
            self.U_iou = nn.Parameter(
                torch.randn(out_channels, 3 * out_channels, self.edge_dim)
            )
        else:
            self.U_iou = nn.Linear(out_channels, 3 * out_channels)
        
        # Forget gate
        self.W_f = nn.Linear(in_channels, out_channels)
        if typed_hidden_rep:
            self.U_f = nn.Parameter(
                torch.randn(out_channels, out_channels, self.edge_dim)
            )
        else:
            self.U_f = nn.Linear(out_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.out_channels, device=device)
        c = torch.zeros(batch_size, self.out_channels, device=device)
        
        # Compute node processing order (topological sort for trees)
        adjacency_list = edge_index.t()
        node_order = torch.zeros(batch_size, device=device)
        for i in range(adjacency_list.shape[0]):
            node_order[adjacency_list[i, 0]] += 1
        
        # Process nodes in topological order
        max_order = int(node_order.max().item()) + 1
        h_sum = torch.zeros(batch_size, self.out_channels, device=device)
        
        for iteration in range(max_order):
            node_mask = node_order == iteration
            
            if iteration == 0:
                iou = self.W_iou(x[node_mask])
            else:
                # Aggregate child hidden states
                edge_mask = (node_order[adjacency_list[:, 0]] == iteration)
                if edge_mask.any():
                    parent_idx = adjacency_list[edge_mask, 0]
                    child_idx = adjacency_list[edge_mask, 1]
                    
                    for pidx, cidx in zip(parent_idx, child_idx):
                        h_sum[pidx] += h[cidx]
                    
                    iou = self.W_iou(x[node_mask]) + self.U_iou(h_sum[node_mask])
                else:
                    iou = self.W_iou(x[node_mask])
            
            # Apply LSTM gates
            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)
            
            c[node_mask] = i * u
            
            if iteration > 0 and edge_mask.any():
                # Compute forget gates
                f = torch.sigmoid(self.W_f(x[parent_idx]) + self.U_f(h[child_idx]))
                fc = f * c[child_idx]
                
                for cidx, pidx in enumerate(parent_idx):
                    c[pidx] += fc[cidx]
                    if self.cell_clip is not None:
                        c[pidx] = torch.clamp(c[pidx], -self.cell_clip, self.cell_clip)
            
            h[node_mask] = o * torch.tanh(c[node_mask])
        
        return h


# ============================================================================
# Linear Encoder
# Used by: Simple baseline, fast inference
# ============================================================================

class LinearEncoder(nn.Module):
    """
    Simple linear transformation encoder.
    
    Provides a lightweight baseline or fast embedding projection.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output embedding dimension
        num_layers: Number of linear layers (default: 1)
        dropout: Dropout rate
        activation: Activation function name
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        current_dim = in_channels
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, out_channels))
            layers.append(create_activation(activation))
            layers.append(nn.Dropout(dropout))
            current_dim = out_channels
        
        layers.append(nn.Linear(current_dim, out_channels))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x, edge_index=None, edge_attr=None, **kwargs):
        return self.encoder(x)


# ============================================================================
# Sum Aggregation Encoder
# Used by: Simple graph-level pooling
# ============================================================================

class SumAggregationEncoder(nn.Module):
    """
    Sum aggregation encoder for graph-level representations.
    
    Aggregates node embeddings via summation to create graph embeddings.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output embedding dimension
        hidden_channels: Optional hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        if hidden_channels is not None:
            self.transform = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            self.transform = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index=None, batch=None, **kwargs):
        x = self.transform(x)
        
        if batch is not None:
            # Graph-level aggregation
            from torch_geometric.nn import global_add_pool
            return global_add_pool(x, batch)
        else:
            # Node-level features
            return x


# ============================================================================
# Encoder Factory Functions
# ============================================================================

def create_gin_encoder(**kwargs):
    """Factory function for GIN encoder."""
    return GINEncoder(**kwargs)


def create_glstm_encoder(**kwargs):
    """Factory function for GLSTM encoder."""
    return GLSTMEncoder(**kwargs)


def create_linear_encoder(**kwargs):
    """Factory function for Linear encoder."""
    return LinearEncoder(**kwargs)


def create_sum_aggregation_encoder(**kwargs):
    """Factory function for Sum Aggregation encoder."""
    return SumAggregationEncoder(**kwargs)

