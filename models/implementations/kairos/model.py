"""
Kairos core model implementation.
Adapted from original Kairos repository for standalone use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, Linear


class TimeEncoder(nn.Module):
    """
    Time encoder for temporal graph learning.
    
    Encodes relative time differences into high-dimensional representations.
    """
    
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.out_channels = dimension

    def forward(self, t):
        """
        Encode time differences.
        
        Args:
            t: Time tensor
            
        Returns:
            Time encodings
        """
        # Reshape time for encoding
        t = t.unsqueeze(dim=1) if len(t.shape) == 1 else t
        output = torch.cos(self.w(t))
        return output


class GraphAttentionEmbedding(nn.Module):
    """
    Graph Attention Embedding for temporal provenance graphs.
    
    Uses TransformerConv with edge attributes and temporal encoding.
    """
    
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        
        # Two-layer Transformer convolution
        self.conv = TransformerConv(
            in_channels, out_channels, heads=8,
            dropout=0.0, edge_dim=edge_dim
        )
        self.conv2 = TransformerConv(
            out_channels * 8, out_channels, heads=1, concat=False,
            dropout=0.0, edge_dim=edge_dim
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, last_update, edge_index, t, msg):
        """
        Forward pass through graph attention layers.
        
        Args:
            x: Node features
            last_update: Last update timestamps for nodes
            edge_index: Edge connectivity
            t: Edge timestamps
            msg: Edge messages/features
            
        Returns:
            Updated node embeddings
        """
        # Compute relative time
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        
        # Concatenate time encoding with edge messages
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        
        # Two-layer graph attention
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        return x


class LinkPredictor(nn.Module):
    """
    Link prediction head for Kairos.
    
    Predicts edge properties or anomaly scores based on node embeddings.
    """
    
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        
        # Source and destination transformations
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        # Multi-layer predictor
        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, in_channels // 2),
            nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels // 2, out_channels)
        )

    def forward(self, z_src, z_dst):
        """
        Predict link properties.
        
        Args:
            z_src: Source node embeddings
            z_dst: Destination node embeddings
            
        Returns:
            Link predictions
        """
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)
        h = self.lin_seq(h)
        return h


class KairosModel(nn.Module):
    """
    Complete Kairos model combining all components.
    """
    
    def __init__(self, in_channels=100, out_channels=100, msg_dim=100,
                 num_classes=2, max_nodes=268243):
        super(KairosModel, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.msg_dim = msg_dim
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        
        # Initialize components
        self.time_enc = TimeEncoder(out_channels)
        self.gnn = GraphAttentionEmbedding(
            in_channels, out_channels, msg_dim, self.time_enc
        )
        self.link_predictor = LinkPredictor(out_channels, num_classes)
        
        # Memory for temporal information (optional)
        self.memory_dim = out_channels
    
    def forward(self, x, last_update, edge_index, t, msg):
        """
        Forward pass through Kairos model.
        
        Args:
            x: Node features
            last_update: Last update timestamps
            edge_index: Edge connectivity
            t: Edge timestamps
            msg: Edge messages
            
        Returns:
            Edge predictions
        """
        # Graph embedding
        h = self.gnn(x, last_update, edge_index, t, msg)
        
        # Link prediction
        pred = self.link_predictor(h[edge_index[0]], h[edge_index[1]])
        
        return pred
    
    def get_embeddings(self, x, last_update, edge_index, t, msg):
        """Get node embeddings."""
        return self.gnn(x, last_update, edge_index, t, msg)
