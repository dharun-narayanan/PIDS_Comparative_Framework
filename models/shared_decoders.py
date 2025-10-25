"""
Shared decoder architectures for PIDS models.

This module contains common decoder implementations used across
multiple models for various tasks:
- Edge prediction/classification
- Node classification
- Anomaly detection
- Contrastive learning
- Feature reconstruction

Decoders:
- EdgeDecoder - Edge-level prediction
- NodeDecoder - Node-level classification  
- ContrastiveDecoder - Temporal contrastive learning
- ReconstructionDecoder - Feature reconstruction
- AnomalyDecoder - Anomaly scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# MLP Builder Utility
# ============================================================================

def build_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    dropout: float = 0.1,
    activation: str = 'relu',
    final_activation: Optional[str] = None,
    batch_norm: bool = False
) -> nn.Sequential:
    """
    Build a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        dropout: Dropout rate
        activation: Activation function name
        final_activation: Final layer activation (None for no activation)
        batch_norm: Whether to use batch normalization
        
    Returns:
        Sequential MLP module
    """
    from models.shared_encoders import create_activation
    
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        
        # Add batch norm (except for last layer)
        if batch_norm and i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i+1]))
        
        # Add activation
        if i < len(dims) - 2:
            layers.append(create_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        elif final_activation is not None:
            layers.append(create_activation(final_activation))
    
    return nn.Sequential(*layers)


# ============================================================================
# Edge Decoder
# Used by: MAGIC, Kairos, Orthrus, Continuum_FL
# ============================================================================

class EdgeDecoder(nn.Module):
    """
    Edge-level decoder for edge classification/prediction.
    
    Predicts edge properties (type, existence, anomaly) based on
    source and destination node embeddings.
    
    Args:
        in_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension (number of classes or 1 for binary)
        num_layers: Number of MLP layers
        dropout: Dropout rate
        use_edge_features: Whether to incorporate edge features
        prediction_type: 'classification', 'binary', or 'regression'
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        prediction_type: str = 'classification'
    ):
        super(EdgeDecoder, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_features = use_edge_features
        self.prediction_type = prediction_type
        self.edge_proj = None  # Edge feature projection layer
        
        # Input is concatenation of source and destination embeddings
        mlp_input_dim = in_dim * 2
        if use_edge_features:
            mlp_input_dim += in_dim  # Add space for edge features
        
        # Build MLP layers
        hidden_dims = [hidden_dim] * num_layers
        self.mlp = build_mlp(
            mlp_input_dim,
            hidden_dims,
            out_dim,
            dropout=dropout,
            activation='relu',
            batch_norm=True
        )
        
        # Loss function
        if prediction_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif prediction_type == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
    
    def forward(
        self,
        h_src,
        h_dst,
        edge_features=None,
        edge_labels=None,
        inference=False
    ):
        """
        Forward pass through edge decoder.
        
        Args:
            h_src: Source node embeddings [num_edges, in_dim]
            h_dst: Destination node embeddings [num_edges, in_dim]
            edge_features: Edge features [num_edges, edge_dim] (optional)
            edge_labels: Ground truth labels [num_edges]
            inference: Whether in inference mode
            
        Returns:
            If training: loss scalar
            If inference: predictions [num_edges, out_dim] or [num_edges]
        """
        # Concatenate source and destination embeddings
        h = torch.cat([h_src, h_dst], dim=-1)
        
        if self.use_edge_features and edge_features is not None:
            # Check if edge features need projection
            edge_dim = edge_features.shape[-1]
            if edge_dim != self.in_dim:
                # Create projection layer if not exists
                if self.edge_proj is None or self.edge_proj.in_features != edge_dim:
                    self.edge_proj = nn.Linear(edge_dim, self.in_dim).to(edge_features.device)
                # Project edge features to expected dimension
                edge_features = self.edge_proj(edge_features)
            
            h = torch.cat([h, edge_features], dim=-1)
        
        # Apply MLP
        logits = self.mlp(h)
        
        if inference:
            # Return predictions
            if self.prediction_type == 'classification':
                if self.out_dim == 2:
                    return F.softmax(logits, dim=-1)[:, 1]  # Anomaly probability
                else:
                    return F.softmax(logits, dim=-1)
            elif self.prediction_type == 'binary':
                return torch.sigmoid(logits).squeeze()
            else:
                return logits
        else:
            # Compute loss
            if edge_labels is not None:
                if self.prediction_type == 'binary':
                    return self.criterion(logits.squeeze(), edge_labels.float())
                else:
                    return self.criterion(logits, edge_labels)
            else:
                return torch.tensor(0.0, device=h.device, requires_grad=True)


# ============================================================================
# Node Decoder
# Used by: Orthrus, ThreaTrace
# ============================================================================

class NodeDecoder(nn.Module):
    """
    Node-level decoder for node classification.
    
    Predicts node properties (type, anomaly, cluster) based on
    node embeddings.
    
    Args:
        in_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension (number of classes)
        num_layers: Number of MLP layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(NodeDecoder, self).__init__()
        
        hidden_dims = [hidden_dim] * num_layers
        self.mlp = build_mlp(
            in_dim,
            hidden_dims,
            out_dim,
            dropout=dropout,
            activation='relu',
            batch_norm=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, h, node_labels=None, inference=False):
        """
        Forward pass through node decoder.
        
        Args:
            h: Node embeddings [num_nodes, in_dim]
            node_labels: Ground truth labels [num_nodes]
            inference: Whether in inference mode
            
        Returns:
            If training: loss scalar
            If inference: predictions [num_nodes, out_dim]
        """
        logits = self.mlp(h)
        
        if inference:
            return F.softmax(logits, dim=-1)
        else:
            if node_labels is not None:
                return self.criterion(logits, node_labels)
            else:
                return torch.tensor(0.0, device=h.device, requires_grad=True)


# ============================================================================
# Contrastive Decoder
# Used by: Orthrus, Kairos (temporal contrastive learning)
# ============================================================================

class ContrastiveDecoder(nn.Module):
    """
    Contrastive learning decoder for temporal graphs.
    
    Learns embeddings by contrasting current representations with
    previous time steps (positive) vs random nodes (negative).
    
    Args:
        temperature: Temperature for contrastive loss
        similarity: Similarity function ('cosine' or 'dot')
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        similarity: str = 'cosine'
    ):
        super(ContrastiveDecoder, self).__init__()
        self.temperature = temperature
        self.similarity = similarity
    
    def compute_similarity(self, h1, h2):
        """Compute pairwise similarity."""
        if self.similarity == 'cosine':
            h1_norm = F.normalize(h1, p=2, dim=-1)
            h2_norm = F.normalize(h2, p=2, dim=-1)
            return torch.mm(h1_norm, h2_norm.t())
        else:  # dot product
            return torch.mm(h1, h2.t())
    
    def forward(
        self,
        h_current,
        h_previous=None,
        positive_mask=None,
        inference=False
    ):
        """
        Forward pass through contrastive decoder.
        
        Args:
            h_current: Current embeddings [num_nodes, dim]
            h_previous: Previous embeddings [num_nodes, dim]
            positive_mask: Mask indicating positive pairs [num_nodes, num_nodes]
            inference: Whether in inference mode
            
        Returns:
            If training: contrastive loss
            If inference: similarity scores
        """
        if inference:
            # Return similarity scores
            if h_previous is not None:
                return self.compute_similarity(h_current, h_previous)
            else:
                return torch.zeros(h_current.size(0), h_current.size(0), device=h_current.device)
        
        if h_previous is None:
            return torch.tensor(0.0, device=h_current.device, requires_grad=True)
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity(h_current, h_previous) / self.temperature
        
        # Create labels for contrastive loss
        if positive_mask is None:
            # Default: diagonal as positive pairs
            positive_mask = torch.eye(h_current.size(0), device=h_current.device).bool()
        
        # Compute contrastive loss
        # Positive pairs should have high similarity, negatives low
        pos_sim = sim_matrix[positive_mask]
        
        # Use InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = pos_sim - torch.log(exp_sim.sum(dim=1, keepdim=True)[positive_mask])
        loss = -log_prob.mean()
        
        return loss


# ============================================================================
# Reconstruction Decoder
# Used by: MAGIC, Continuum_FL (autoencoder-based models)
# ============================================================================

class ReconstructionDecoder(nn.Module):
    """
    Feature reconstruction decoder for autoencoders.
    
    Reconstructs node/edge features from learned embeddings.
    Used for unsupervised pre-training and anomaly detection.
    
    Args:
        in_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output feature dimension
        num_layers: Number of MLP layers
        dropout: Dropout rate
        loss_type: Loss function ('mse', 'mae', or 'cosine')
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        loss_type: str = 'mse'
    ):
        super(ReconstructionDecoder, self).__init__()
        
        hidden_dims = [hidden_dim] * num_layers
        self.mlp = build_mlp(
            in_dim,
            hidden_dims,
            out_dim,
            dropout=dropout,
            activation='relu'
        )
        
        # Loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            self.criterion = nn.MSELoss()
        
        self.loss_type = loss_type
    
    def forward(self, h, target_features=None, inference=False):
        """
        Forward pass through reconstruction decoder.
        
        Args:
            h: Embeddings [num_items, in_dim]
            target_features: Target features to reconstruct [num_items, out_dim]
            inference: Whether in inference mode
            
        Returns:
            If training: reconstruction loss
            If inference: reconstructed features [num_items, out_dim]
        """
        reconstructed = self.mlp(h)
        
        if inference:
            return reconstructed
        else:
            if target_features is not None:
                if self.loss_type == 'cosine':
                    # Cosine similarity expects target of 1 (similar) or -1 (dissimilar)
                    target = torch.ones(h.size(0), device=h.device)
                    return self.criterion(reconstructed, target_features, target)
                else:
                    return self.criterion(reconstructed, target_features)
            else:
                return torch.tensor(0.0, device=h.device, requires_grad=True)


# ============================================================================
# Anomaly Decoder
# Used by: All models for anomaly detection
# ============================================================================

class AnomalyDecoder(nn.Module):
    """
    Anomaly scoring decoder.
    
    Computes anomaly scores for nodes or edges based on embeddings.
    Can use reconstruction error, classification, or distance-based scoring.
    
    Args:
        in_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        dropout: Dropout rate
        scoring_method: 'reconstruction', 'classification', or 'distance'
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        scoring_method: str = 'classification'
    ):
        super(AnomalyDecoder, self).__init__()
        
        self.scoring_method = scoring_method
        
        if scoring_method == 'classification':
            # Binary classification decoder
            hidden_dims = [hidden_dim] * num_layers
            self.mlp = build_mlp(
                in_dim,
                hidden_dims,
                1,  # Binary output
                dropout=dropout,
                activation='relu',
                final_activation='sigmoid'
            )
            self.criterion = nn.BCELoss()
        
        elif scoring_method == 'reconstruction':
            # Reconstruction-based anomaly detection
            self.decoder = ReconstructionDecoder(
                in_dim,
                hidden_dim,
                in_dim,
                num_layers,
                dropout
            )
        
        elif scoring_method == 'distance':
            # Distance from normal embeddings
            self.register_buffer('normal_centroid', torch.zeros(in_dim))
            self.mlp = build_mlp(
                in_dim,
                [hidden_dim],
                in_dim,
                dropout=dropout,
                activation='relu'
            )
    
    def forward(self, h, labels=None, features=None, inference=False):
        """
        Forward pass through anomaly decoder.
        
        Args:
            h: Embeddings [num_items, in_dim]
            labels: Binary anomaly labels [num_items] (optional)
            features: Original features for reconstruction (optional)
            inference: Whether in inference mode
            
        Returns:
            If training: loss
            If inference: anomaly scores [num_items]
        """
        if self.scoring_method == 'classification':
            scores = self.mlp(h).squeeze()
            
            if inference:
                return scores
            else:
                if labels is not None:
                    return self.criterion(scores, labels.float())
                else:
                    return torch.tensor(0.0, device=h.device, requires_grad=True)
        
        elif self.scoring_method == 'reconstruction':
            if inference:
                reconstructed = self.decoder(h, inference=True)
                if features is not None:
                    # Compute reconstruction error as anomaly score
                    scores = torch.norm(reconstructed - features, dim=-1)
                    return scores
                else:
                    return torch.zeros(h.size(0), device=h.device)
            else:
                return self.decoder(h, features, inference=False)
        
        elif self.scoring_method == 'distance':
            h_projected = self.mlp(h)
            
            if inference:
                # Distance from normal centroid
                scores = torch.norm(h_projected - self.normal_centroid, dim=-1)
                return scores
            else:
                # Update centroid with normal samples
                if labels is not None:
                    normal_mask = labels == 0
                    if normal_mask.any():
                        normal_embeddings = h_projected[normal_mask]
                        self.normal_centroid = normal_embeddings.mean(dim=0).detach()
                return torch.tensor(0.0, device=h.device, requires_grad=True)


# ============================================================================
# Inner Product Decoder
# Used by: Simple link prediction models
# ============================================================================

class InnerProductDecoder(nn.Module):
    """
    Simple inner product decoder for link prediction.
    
    Computes edge scores as dot product of node embeddings.
    
    Args:
        dropout: Dropout rate
    """
    
    def __init__(self, dropout: float = 0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_src, h_dst, edge_labels=None, inference=False):
        """
        Compute edge scores using inner product.
        
        Args:
            h_src: Source node embeddings
            h_dst: Destination node embeddings
            edge_labels: Ground truth labels
            inference: Whether in inference mode
            
        Returns:
            Edge scores or loss
        """
        h_src = self.dropout(h_src)
        h_dst = self.dropout(h_dst)
        
        scores = (h_src * h_dst).sum(dim=-1)
        
        if inference:
            return torch.sigmoid(scores)
        else:
            if edge_labels is not None:
                loss = F.binary_cross_entropy_with_logits(scores, edge_labels.float())
                return loss
            else:
                return torch.tensor(0.0, device=scores.device, requires_grad=True)


# ============================================================================
# Decoder Factory
# ============================================================================

def get_decoder(decoder_type: str, config: dict) -> nn.Module:
    """
    Factory function to create decoder based on type and configuration.
    
    Args:
        decoder_type: Type of decoder
        config: Configuration dictionary
        
    Returns:
        Decoder module
    """
    decoder_type = decoder_type.lower()
    
    if decoder_type in ['edge', 'edge_decoder']:
        return EdgeDecoder(**config)
    
    elif decoder_type in ['node', 'node_decoder']:
        return NodeDecoder(**config)
    
    elif decoder_type in ['contrastive', 'contrastive_decoder']:
        return ContrastiveDecoder(**config)
    
    elif decoder_type in ['reconstruction', 'autoencoder']:
        return ReconstructionDecoder(**config)
    
    elif decoder_type in ['anomaly', 'anomaly_decoder']:
        return AnomalyDecoder(**config)
    
    elif decoder_type in ['inner_product', 'dot_product']:
        return InnerProductDecoder(**config)
    
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
