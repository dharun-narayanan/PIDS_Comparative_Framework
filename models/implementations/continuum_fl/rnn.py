"""
RNN Cells for Temporal Aggregation in Continuum_FL
"""

import torch.nn as nn
from torch.nn import GRUCell


class RNN_Cells(nn.Module):
    """
    Recurrent Neural Network Cells for temporal snapshot processing
    
    Uses GRU cells to aggregate information across temporal graph snapshots.
    Each cell processes one temporal snapshot, passing hidden state to the next.
    """
    
    def __init__(self, input_dim, hidden_dim, n_cells, device):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            n_cells: Number of RNN cells (should match number of snapshots)
            device: Device to place the model on
        """
        super(RNN_Cells, self).__init__()
        self.cells = nn.ModuleList()
        
        for i in range(n_cells):
            self.cells.append(GRUCell(input_dim, hidden_dim, device=device))

    def forward(self, inputs):
        """
        Process temporal snapshots sequentially
        
        Args:
            inputs: List of input tensors, one per temporal snapshot
            
        Returns:
            results: List of hidden states, one per snapshot
        """
        results = []
        for i in range(len(self.cells)):
            if i == 0:
                # First snapshot: initialize hidden state with input
                results.append(self.cells[i](inputs[i], inputs[i]))
            else:
                # Subsequent snapshots: use previous hidden state
                results.append(self.cells[i](inputs[i], results[i-1]))

        return results
