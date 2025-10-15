"""
Time encoder for Kairos model.
"""

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Time encoder for temporal graph learning.
    
    Encodes relative time differences into high-dimensional representations
    using learnable periodic functions.
    
    Args:
        dimension: Output dimension for time encodings
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
            t: Time tensor (relative time differences)
            
        Returns:
            Time encodings of shape [*, dimension]
        """
        # Ensure correct shape
        if len(t.shape) == 1:
            t = t.unsqueeze(dim=1)
        
        # Apply linear transformation and cosine activation
        output = torch.cos(self.w(t))
        return output
