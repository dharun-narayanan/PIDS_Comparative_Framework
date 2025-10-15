"""
Encoders for Orthrus model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv, GATConv


class GraphTransformer(nn.Module):
    """
    Graph Transformer encoder using TransformerConv layers.
    
    Args:
        in_dim: Input feature dimension
        hid_dim: Hidden dimension
        out_dim: Output dimension
        edge_dim: Edge feature dimension
        dropout: Dropout rate
        activation: Activation function
        num_heads: Number of attention heads
    """
    
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, dropout, activation, num_heads):
        super(GraphTransformer, self).__init__()
        
        self.conv = TransformerConv(in_dim, hid_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hid_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        """Forward pass through graph transformer."""
        x = self.activation(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_feats)
        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE encoder.
    
    Args:
        in_dim: Input feature dimension
        hid_dim: Hidden dimension
        out_dim: Output dimension
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(self, in_dim, hid_dim, out_dim, dropout, activation):
        super(GraphSAGE, self).__init__()
        
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x, edge_index, **kwargs):
        """Forward pass through GraphSAGE."""
        x = self.activation(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GraphAttention(nn.Module):
    """
    Graph Attention Network encoder.
    
    Args:
        in_dim: Input feature dimension
        hid_dim: Hidden dimension
        out_dim: Output dimension
        dropout: Dropout rate
        activation: Activation function
        num_heads: Number of attention heads
    """
    
    def __init__(self, in_dim, hid_dim, out_dim, dropout, activation, num_heads):
        super(GraphAttention, self).__init__()
        
        self.conv1 = GATConv(in_dim, hid_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hid_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x, edge_index, **kwargs):
        """Forward pass through GAT."""
        x = self.activation(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def get_encoder(config):
    """
    Factory function to create encoder based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Encoder module
    """
    encoder_type = config.get('encoder_type', 'transformer')
    in_dim = config.get('in_dim', 100)
    hid_dim = config.get('hid_dim', 256)
    out_dim = config.get('out_dim', 100)
    edge_dim = config.get('edge_dim', 64)
    dropout = config.get('dropout', 0.1)
    num_heads = config.get('num_heads', 8)
    
    # Create activation
    activation_name = config.get('activation', 'relu')
    if activation_name == 'relu':
        activation = nn.ReLU()
    elif activation_name == 'prelu':
        activation = nn.PReLU()
    elif activation_name == 'elu':
        activation = nn.ELU()
    else:
        activation = nn.ReLU()
    
    if encoder_type == 'transformer':
        return GraphTransformer(
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            edge_dim=edge_dim,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads
        )
    elif encoder_type == 'sage':
        return GraphSAGE(
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation
        )
    elif encoder_type == 'gat':
        return GraphAttention(
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
