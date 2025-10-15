"""
Decoders for Orthrus model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDecoder(nn.Module):
    """
    Edge prediction decoder.
    
    Predicts edge properties based on node embeddings.
    
    Args:
        in_dim: Input embedding dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension (number of classes)
        dropout: Dropout rate
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(EdgeDecoder, self).__init__()
        
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_src, h_dst, x, edge_index, edge_type=None, inference=False, **kwargs):
        """
        Forward pass through edge decoder.
        
        Args:
            h_src: Source node embeddings
            h_dst: Destination node embeddings
            x: Node features (not used here)
            edge_index: Edge connectivity
            edge_type: Edge type labels
            inference: Whether in inference mode
            
        Returns:
            Loss (training) or scores (inference)
        """
        # Concatenate source and destination embeddings
        h = torch.cat([h_src, h_dst], dim=-1)
        
        # Apply MLP
        h = self.activation(self.fc1(h))
        h = self.dropout(h)
        h = self.activation(self.fc2(h))
        h = self.dropout(h)
        h = self.fc3(h)
        
        if inference:
            # Return edge scores
            return F.softmax(h, dim=-1)[:, 1] if h.size(1) == 2 else h
        else:
            # Compute loss
            if edge_type is not None:
                loss = self.criterion(h, edge_type)
            else:
                # Default: predict all edges as positive
                loss = torch.tensor(0.0, device=h.device)
            return loss


class ContrastiveDecoder(nn.Module):
    """
    Contrastive learning decoder.
    
    Uses temporal contrastive learning to improve embeddings.
    
    Args:
        temperature: Temperature for contrastive loss
    """
    
    def __init__(self, temperature=0.07):
        super(ContrastiveDecoder, self).__init__()
        self.temperature = temperature

    def forward(self, h_src, h_dst, x, edge_index, inference=False, 
                last_h_storage=None, last_h_non_empty_nodes=None, **kwargs):
        """
        Forward pass through contrastive decoder.
        
        Args:
            h_src: Source node embeddings
            h_dst: Destination node embeddings
            x: Node features
            edge_index: Edge connectivity
            inference: Whether in inference mode
            last_h_storage: Storage of previous embeddings
            last_h_non_empty_nodes: Nodes with previous embeddings
            
        Returns:
            Loss (training) or zeros (inference)
        """
        if inference or last_h_storage is None:
            return torch.tensor(0.0, device=h_src.device)
        
        # Compute contrastive loss
        # Positive pairs: current and previous embeddings of same node
        src, dst = edge_index
        involved_nodes = torch.cat([src, dst]).unique()
        
        # Filter nodes that have previous embeddings
        valid_nodes = torch.isin(involved_nodes, last_h_non_empty_nodes)
        if valid_nodes.sum() == 0:
            return torch.tensor(0.0, device=h_src.device)
        
        valid_nodes = involved_nodes[valid_nodes]
        
        # Get current and previous embeddings
        curr_h = torch.cat([h_src, h_dst], dim=0)
        curr_h = curr_h[torch.isin(torch.cat([src, dst]), valid_nodes)]
        prev_h = last_h_storage[valid_nodes]
        
        if curr_h.size(0) == 0:
            return torch.tensor(0.0, device=h_src.device)
        
        # Compute similarity
        curr_h = F.normalize(curr_h, dim=-1)
        prev_h = F.normalize(prev_h, dim=-1)
        
        similarity = torch.mm(curr_h, prev_h.t()) / self.temperature
        labels = torch.arange(curr_h.size(0), device=h_src.device)
        
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class AttributeDecoder(nn.Module):
    """
    Node attribute prediction decoder.
    
    Reconstructs node attributes from embeddings.
    
    Args:
        in_dim: Input embedding dimension
        out_dim: Output attribute dimension
    """
    
    def __init__(self, in_dim, out_dim):
        super(AttributeDecoder, self).__init__()
        
        self.fc = nn.Linear(in_dim, out_dim)
        self.criterion = nn.MSELoss()

    def forward(self, h_src, h_dst, x, inference=False, **kwargs):
        """
        Forward pass through attribute decoder.
        
        Args:
            h_src: Source node embeddings
            h_dst: Destination node embeddings
            x: Original node features
            inference: Whether in inference mode
            
        Returns:
            Loss (training) or reconstructed attributes (inference)
        """
        x_src, x_dst = x
        
        # Reconstruct attributes
        x_src_recon = self.fc(h_src)
        x_dst_recon = self.fc(h_dst)
        
        if inference:
            return torch.cat([x_src_recon, x_dst_recon], dim=0)
        else:
            # Compute reconstruction loss
            loss_src = self.criterion(x_src_recon, x_src)
            loss_dst = self.criterion(x_dst_recon, x_dst)
            return (loss_src + loss_dst) / 2


def get_decoders(config):
    """
    Factory function to create decoders based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of decoder modules
    """
    decoders = []
    decoder_types = config.get('decoder_types', ['edge'])
    
    in_dim = config.get('out_dim', 100)
    hidden_dim = config.get('decoder_hidden_dim', 256)
    num_classes = config.get('num_classes', 2)
    dropout = config.get('dropout', 0.1)
    
    for decoder_type in decoder_types:
        if decoder_type == 'edge':
            decoders.append(EdgeDecoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=num_classes,
                dropout=dropout
            ))
        elif decoder_type == 'contrastive':
            temperature = config.get('contrastive_temperature', 0.07)
            decoders.append(ContrastiveDecoder(temperature=temperature))
        elif decoder_type == 'attribute':
            attr_dim = config.get('in_dim', 100)
            decoders.append(AttributeDecoder(in_dim=in_dim, out_dim=attr_dim))
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    return decoders
