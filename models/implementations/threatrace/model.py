"""
ThreaTrace Model - Sketch-based Provenance Graph Anomaly Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class ThreaTraceModel(nn.Module):
    """
    ThreaTrace: Sketch-based provenance graph anomaly detection
    
    Uses GraphSAGE or GAT for node embeddings with sketch-based
    fixed-size graph representation for efficient anomaly detection.
    
    Architecture:
    - Graph encoder (SAGE or GAT)
    - Sketch generator for fixed-size representation
    - Classification head
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 encoder_type='sage', num_layers=2, dropout=0.5, 
                 sketch_size=256, num_classes=2):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            encoder_type: Type of encoder ('sage' or 'gat')
            num_layers: Number of GNN layers
            dropout: Dropout rate
            sketch_size: Size of sketch representation
            num_classes: Number of output classes
        """
        super(ThreaTraceModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.sketch_size = sketch_size
        
        # Build encoder layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        if encoder_type == 'sage':
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=False))
        elif encoder_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
            hidden_channels = hidden_channels * 4  # Account for multi-head concatenation
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if encoder_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            elif encoder_type == 'gat':
                self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if encoder_type == 'sage':
            self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        elif encoder_type == 'gat':
            self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))
        
        # Sketch projection
        self.sketch_proj = nn.Linear(out_channels, sketch_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(sketch_size, sketch_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sketch_size // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes (optional)
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Encode graph
        embeddings = self.encode(x, edge_index)
        
        # Generate sketch representation
        sketch = self.sketch_proj(embeddings)
        
        # Pool to graph-level representation
        if batch is not None:
            # Batch-wise pooling
            sketch = self.global_mean_pool(sketch, batch)
        else:
            # Single graph pooling
            sketch = sketch.mean(dim=0, keepdim=True)
        
        # Classify
        logits = self.classifier(sketch)
        
        return logits
    
    def encode(self, x, edge_index):
        """
        Encode graph into node embeddings
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            embeddings: Node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # No BN/ReLU/dropout on last layer
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def global_mean_pool(self, x, batch):
        """
        Global mean pooling across batch
        
        Args:
            x: Node features [num_nodes, feature_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            pooled: Graph-level features [batch_size, feature_dim]
        """
        batch_size = batch.max().item() + 1
        pooled = torch.zeros(batch_size, x.size(1), device=x.device)
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                pooled[i] = x[mask].mean(dim=0)
        
        return pooled
    
    def get_sketch(self, x, edge_index, batch=None):
        """
        Get sketch representation without classification
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment (optional)
            
        Returns:
            sketch: Graph sketch representation
        """
        embeddings = self.encode(x, edge_index)
        sketch = self.sketch_proj(embeddings)
        
        if batch is not None:
            sketch = self.global_mean_pool(sketch, batch)
        else:
            sketch = sketch.mean(dim=0, keepdim=True)
        
        return sketch


class SAGENet(nn.Module):
    """
    GraphSAGE-based network for ThreaTrace
    
    Simplified interface for backward compatibility.
    """
    
    def __init__(self, in_channels, out_channels, hidden_channels=32, 
                 concat=False, dropout=0.5):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=False, concat=concat)
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=False, concat=concat)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            log_probs: Log probabilities
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
