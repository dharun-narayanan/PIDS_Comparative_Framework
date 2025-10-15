"""
Orthrus core model implementation.
Adapted from original Orthrus repository for standalone use.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Orthrus(nn.Module):
    """
    Orthrus: Provenance-based IDS with high-quality attribution.
    
    Combines encoder and multiple decoders for multi-task learning
    with optional contrastive learning.
    
    Args:
        encoder: Graph encoder module
        decoders: List of decoder modules for different tasks
        num_nodes: Total number of nodes in the graph
        in_dim: Input feature dimension
        out_dim: Output embedding dimension
        use_contrastive_learning: Whether to use contrastive learning
        device: Device to run on
        graph_reindexer: Graph reindexing utility (optional)
    """
    
    def __init__(self,
                 encoder: nn.Module,
                 decoders: list,
                 num_nodes: int,
                 in_dim: int,
                 out_dim: int,
                 use_contrastive_learning: bool,
                 device,
                 graph_reindexer=None):
        super(Orthrus, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        
        # Storage for contrastive learning
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
        
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
    def forward(self, batch, full_data=None, inference=False):
        """
        Forward pass through Orthrus.
        
        Args:
            batch: Batch of graph data
            full_data: Full dataset for neighbor sampling
            inference: Whether in inference mode
            
        Returns:
            Loss (training) or scores (inference)
        """
        train_mode = not inference
        
        # Extract batch components
        x_src = batch.x_src if hasattr(batch, 'x_src') else batch.x
        x_dst = batch.x_dst if hasattr(batch, 'x_dst') else batch.x
        x = (x_src, x_dst)
        edge_index = batch.edge_index

        with torch.set_grad_enabled(train_mode):
            # Encode
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t if hasattr(batch, 't') else None,
                x=x,
                msg=batch.msg if hasattr(batch, 'msg') else None,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data,
                inference=inference,
                edge_types=batch.edge_type if hasattr(batch, 'edge_type') else None
            )

            # Handle encoder output
            if isinstance(h, torch.Tensor):
                h_src, h_dst = h[edge_index[0]], h[edge_index[1]]
            else:
                h_src, h_dst = h
        
            # Adjust x if needed
            if x[0].shape[0] != edge_index.shape[1]:
                x = (x_src[edge_index[0]], x_dst[edge_index[1]])
            
            # Update contrastive learning storage
            if self.use_contrastive_learning:
                involved_nodes = edge_index.flatten()
                self.last_h_storage[involved_nodes] = torch.cat([h_src, h_dst]).detach()
                self.last_h_non_empty_nodes = torch.cat([involved_nodes, self.last_h_non_empty_nodes]).unique()
            
            # Initialize loss/scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(edge_index.shape[1], dtype=torch.float)).to(h_src.device)
            
            # Apply decoders
            for decoder in self.decoders:
                loss = decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    x=x,
                    edge_index=edge_index,
                    edge_type=batch.edge_type if hasattr(batch, 'edge_type') else None,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                )
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")
                loss_or_scores = loss_or_scores + loss

            return loss_or_scores
    
    def get_embeddings(self, batch, full_data=None):
        """Get node embeddings."""
        with torch.no_grad():
            x = (batch.x_src if hasattr(batch, 'x_src') else batch.x,
                 batch.x_dst if hasattr(batch, 'x_dst') else batch.x)
            
            h = self.encoder(
                edge_index=batch.edge_index,
                t=batch.t if hasattr(batch, 't') else None,
                x=x,
                msg=batch.msg if hasattr(batch, 'msg') else None,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data,
                inference=True,
                edge_types=batch.edge_type if hasattr(batch, 'edge_type') else None
            )
            
            if isinstance(h, torch.Tensor):
                return h
            else:
                return torch.cat(h, dim=0)
    
    def reset_state(self):
        """Reset model state (for temporal models)."""
        if hasattr(self.encoder, 'reset_state'):
            self.encoder.reset_state()
        if self.use_contrastive_learning:
            self.last_h_storage.zero_()
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=self.device)


class OrthrusEncoder(nn.Module):
    """
    Orthrus encoder with neighbor sampling and temporal features.
    
    Args:
        encoder: Base encoder module (e.g., GraphTransformer)
        neighbor_loader: Neighbor sampling loader
        in_dim: Input feature dimension
        temporal_dim: Temporal feature dimension
        use_node_feats_in_gnn: Whether to use node features in GNN
        graph_reindexer: Graph reindexing utility
        edge_features: List of edge feature types to use
        device: Device to run on
        num_nodes: Total number of nodes
        edge_dim: Edge feature dimension
    """
    
    def __init__(self,
                 encoder,
                 neighbor_loader,
                 in_dim,
                 temporal_dim,
                 use_node_feats_in_gnn,
                 graph_reindexer,
                 edge_features,
                 device,
                 num_nodes,
                 edge_dim):
        super(OrthrusEncoder, self).__init__()
        self.encoder = encoder
        self.neighbor_loader = neighbor_loader
        self.device = device
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        
        self.edge_features = edge_features

        self.use_node_feats_in_gnn = use_node_feats_in_gnn
        if self.use_node_feats_in_gnn:
            self.src_linear = nn.Linear(in_dim, temporal_dim)
            self.dst_linear = nn.Linear(in_dim, temporal_dim)
            
        self.graph_reindexer = graph_reindexer
        self.num_nodes = num_nodes
        self.temporal_dim = temporal_dim

    def forward(self, edge_index, t, msg, x, full_data=None, inference=False, edge_types=None, **kwargs):
        """
        Forward pass through encoder.
        
        Args:
            edge_index: Edge connectivity
            t: Timestamps
            msg: Edge messages
            x: Node features (tuple of src, dst)
            full_data: Full dataset
            inference: Whether in inference mode
            edge_types: Edge types
            
        Returns:
            Tuple of (h_src, h_dst) embeddings
        """
        src, dst = edge_index
        x_src, x_dst = x
        batch_edge_index = edge_index.clone()
        
        # Neighbor sampling
        n_id = torch.cat([src, dst]).unique()
        
        if self.neighbor_loader is not None:
            n_id, edge_index, e_id = self.neighbor_loader(n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        else:
            e_id = torch.arange(edge_index.shape[1], device=self.device)
            self.assoc[n_id] = n_id

        # Project node features
        x_proj = None
        if self.use_node_feats_in_gnn:
            if self.graph_reindexer is not None:
                x_src, x_dst = self.graph_reindexer.node_features_reshape(
                    batch_edge_index, x_src, x_dst, max_num_node=n_id.max()
                )
            x_proj = self.src_linear(x_src[n_id]) + self.dst_linear(x_dst[n_id])
        else:
            # Use zero features if not using node features
            x_proj = torch.zeros(n_id.size(0), self.temporal_dim, device=self.device)

        h = x_proj
        
        # Edge features
        edge_feats = []
        if full_data is not None:
            if "edge_type" in self.edge_features and hasattr(full_data, 'edge_type'):
                curr_msg = full_data.edge_type[e_id.cpu()].to(self.device)
                edge_feats.append(curr_msg)
            if "msg" in self.edge_features and hasattr(full_data, 'msg'):
                curr_msg = full_data.msg[e_id.cpu()].to(self.device)
                edge_feats.append(curr_msg)
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
        
        # Apply encoder
        h = self.encoder(h, edge_index, edge_feats=edge_feats)

        # Extract embeddings for batch edges
        h_src = h[self.assoc[src]]
        h_dst = h[self.assoc[dst]]

        # Update neighbor loader
        if self.neighbor_loader is not None and hasattr(self.neighbor_loader, 'insert'):
            self.neighbor_loader.insert(src, dst)
        
        return h_src, h_dst

    def reset_state(self):
        """Reset encoder state."""
        if self.neighbor_loader is not None and hasattr(self.neighbor_loader, 'reset_state'):
            self.neighbor_loader.reset_state()
