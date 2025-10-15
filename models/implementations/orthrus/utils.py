"""
Utility functions for Orthrus model.
"""

import torch
import torch.nn as nn


def setup_orthrus_model(config):
    """
    Setup Orthrus model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Orthrus model
    """
    from .model import Orthrus, OrthrusEncoder
    from .encoders import get_encoder
    from .decoders import get_decoders
    
    # Get base encoder
    base_encoder = get_encoder(config)
    
    # Create Orthrus encoder (with neighbor sampling if needed)
    if config.get('use_neighbor_sampling', False):
        encoder = OrthrusEncoder(
            encoder=base_encoder,
            neighbor_loader=None,  # Will be set externally
            in_dim=config.get('in_dim', 100),
            temporal_dim=config.get('out_dim', 100),
            use_node_feats_in_gnn=config.get('use_node_feats_in_gnn', True),
            graph_reindexer=None,  # Will be set externally
            edge_features=config.get('edge_features', ['edge_type', 'msg']),
            device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            num_nodes=config.get('num_nodes', 100000),
            edge_dim=config.get('edge_dim', 64)
        )
    else:
        encoder = base_encoder
    
    # Get decoders
    decoders = get_decoders(config)
    
    # Create Orthrus model
    model = Orthrus(
        encoder=encoder,
        decoders=decoders,
        num_nodes=config.get('num_nodes', 100000),
        in_dim=config.get('in_dim', 100),
        out_dim=config.get('out_dim', 100),
        use_contrastive_learning=config.get('use_contrastive_learning', True),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        graph_reindexer=None
    )
    
    return model


class SimpleNeighborLoader:
    """
    Simple neighbor loader for Orthrus.
    
    This is a simplified version for standalone use.
    """
    
    def __init__(self, num_neighbors=10):
        self.num_neighbors = num_neighbors
        self.history = []
    
    def __call__(self, n_id):
        """
        Sample neighbors for given nodes.
        
        Args:
            n_id: Node IDs
            
        Returns:
            Tuple of (sampled_nodes, edge_index, edge_ids)
        """
        # Simplified: just return the input nodes
        # In full implementation, this would sample k-hop neighbors
        num_nodes = n_id.size(0)
        edge_index = torch.stack([
            torch.arange(num_nodes, device=n_id.device),
            torch.arange(num_nodes, device=n_id.device)
        ])
        e_id = torch.arange(edge_index.shape[1], device=n_id.device)
        
        return n_id, edge_index, e_id
    
    def insert(self, src, dst):
        """Insert edges into history."""
        self.history.append((src, dst))
    
    def reset_state(self):
        """Reset loader state."""
        self.history = []


class GraphReindexer:
    """
    Graph reindexing utility for Orthrus.
    
    Handles node feature reshaping and reindexing.
    """
    
    def __init__(self):
        pass
    
    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node):
        """
        Reshape node features to accommodate all nodes.
        
        Args:
            edge_index: Edge connectivity
            x_src: Source node features
            x_dst: Destination node features
            max_num_node: Maximum node ID
            
        Returns:
            Reshaped (x_src, x_dst)
        """
        # Ensure features are sized correctly
        if x_src.size(0) < max_num_node + 1:
            # Pad with zeros
            pad_size = max_num_node + 1 - x_src.size(0)
            x_src = torch.cat([
                x_src,
                torch.zeros(pad_size, x_src.size(1), device=x_src.device)
            ], dim=0)
            x_dst = torch.cat([
                x_dst,
                torch.zeros(pad_size, x_dst.size(1), device=x_dst.device)
            ], dim=0)
        
        return x_src, x_dst


def prepare_orthrus_batch(batch, device):
    """
    Prepare batch data for Orthrus model.
    
    Args:
        batch: Input batch
        device: Target device
        
    Returns:
        Prepared batch
    """
    # Move batch to device
    if hasattr(batch, 'to'):
        batch = batch.to(device)
    
    # Ensure required attributes exist
    if not hasattr(batch, 'x_src'):
        if hasattr(batch, 'x'):
            batch.x_src = batch.x
            batch.x_dst = batch.x
        else:
            raise ValueError("Batch must have x_src/x_dst or x attributes")
    
    if not hasattr(batch, 'edge_index'):
        raise ValueError("Batch must have edge_index attribute")
    
    return batch
