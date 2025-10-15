"""
Loss functions for MAGIC model.
"""

import torch
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    """
    Symmetric Cross Entropy Loss.
    
    Args:
        x: First embedding tensor
        y: Second embedding tensor
        alpha: Alpha parameter for loss computation
        
    Returns:
        SCE loss value
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def masked_mse_loss(input, target, mask):
    """
    Masked MSE loss for attribute reconstruction.
    
    Args:
        input: Predicted attributes
        target: Target attributes  
        mask: Mask indicating which nodes to include
        
    Returns:
        Masked MSE loss
    """
    loss = F.mse_loss(input[mask], target[mask], reduction='mean')
    return loss
