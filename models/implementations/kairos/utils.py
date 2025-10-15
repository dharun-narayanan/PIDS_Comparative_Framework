"""
Utility functions for Kairos model.
"""

import torch
import torch.nn as nn


def setup_kairos_model(config):
    """
    Setup Kairos model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Kairos model
    """
    from .model import KairosModel
    
    model = KairosModel(
        in_channels=config.get('in_channels', 100),
        out_channels=config.get('out_channels', 100),
        msg_dim=config.get('msg_dim', 100),
        num_classes=config.get('num_classes', 2),
        max_nodes=config.get('max_nodes', 268243)
    )
    
    return model


def cal_pos_edges_loss_multiclass(link_pred_ratio, labels):
    """
    Calculate multi-class loss for positive edges.
    
    Args:
        link_pred_ratio: Link prediction scores
        labels: Ground truth labels
        
    Returns:
        Loss tensor
    """
    criterion = nn.CrossEntropyLoss()
    loss = []
    
    for i in range(len(link_pred_ratio)):
        loss.append(
            criterion(
                link_pred_ratio[i].reshape(1, -1),
                labels[i].reshape(-1)
            )
        )
    
    return torch.tensor(loss)


def prepare_kairos_batch(batch, device):
    """
    Prepare batch data for Kairos model.
    
    Args:
        batch: Input batch
        device: Target device
        
    Returns:
        Prepared tensors (x, last_update, edge_index, t, msg)
    """
    x = batch.x.to(device) if hasattr(batch, 'x') else None
    last_update = batch.last_update.to(device) if hasattr(batch, 'last_update') else None
    edge_index = batch.edge_index.to(device) if hasattr(batch, 'edge_index') else None
    t = batch.t.to(device) if hasattr(batch, 't') else None
    msg = batch.msg.to(device) if hasattr(batch, 'msg') else None
    
    # Handle missing last_update
    if last_update is None and t is not None and x is not None:
        last_update = torch.zeros(x.size(0), device=device)
    
    return x, last_update, edge_index, t, msg
