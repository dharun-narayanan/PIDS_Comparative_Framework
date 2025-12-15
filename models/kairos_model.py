"""
Kairos Model Implementation - Temporal Graph Attention Network

This implements the Kairos architecture from the paper:
"Kairos: Practical Intrusion Detection and Investigation using Whole-system Provenance"
IEEE S&P 2024

Key components:
- Temporal graph attention with memory
- Time encoding for temporal relationships
- Edge-level anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalMemory(nn.Module):
    """
    Temporal memory module for storing and updating node states over time.
    
    Based on Temporal Graph Networks (TGN) memory mechanism.
    """
    
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        time_dim: int,
        message_dim: int,
        updater_type: str = 'gru'
    ):
        """
        Initialize temporal memory.
        
        Args:
            num_nodes: Maximum number of nodes
            memory_dim: Dimension of memory vectors
            time_dim: Dimension of time encoding
            message_dim: Dimension of messages
            updater_type: Type of memory updater ('gru' or 'lstm')
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.message_dim = message_dim
        
        # Memory storage: [num_nodes, memory_dim]
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update_time', torch.zeros(num_nodes))
        
        # Message aggregation
        self.message_aggregator = nn.Linear(message_dim, memory_dim)
        
        # Memory updater (GRU or LSTM)
        if updater_type == 'gru':
            self.memory_updater = nn.GRUCell(memory_dim + time_dim, memory_dim)
        elif updater_type == 'lstm':
            self.memory_updater = nn.LSTMCell(memory_dim + time_dim, memory_dim)
        else:
            raise ValueError(f"Unknown updater type: {updater_type}")
        
        self.updater_type = updater_type
    
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve memory for specific nodes.
        
        Args:
            node_ids: Node indices [batch_size]
            
        Returns:
            Memory vectors [batch_size, memory_dim]
        """
        return self.memory[node_ids]
    
    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
        time_encoding: torch.Tensor
    ):
        """
        Update memory for nodes based on new messages.
        
        Args:
            node_ids: Node indices [batch_size]
            messages: Messages to aggregate [batch_size, message_dim]
            timestamps: Event timestamps [batch_size]
            time_encoding: Encoded time features [batch_size, time_dim]
        """
        # Get current memory
        current_memory = self.memory[node_ids]  # [batch_size, memory_dim]
        
        # Aggregate messages
        aggregated_messages = self.message_aggregator(messages)  # [batch_size, memory_dim]
        
        # Combine with time encoding
        update_input = torch.cat([aggregated_messages, time_encoding], dim=-1)
        
        # Update memory
        if self.updater_type == 'gru':
            new_memory = self.memory_updater(update_input, current_memory)
        else:  # LSTM
            new_memory, _ = self.memory_updater(update_input, (current_memory, current_memory))
        
        # Update storage
        self.memory[node_ids] = new_memory
        self.last_update_time[node_ids] = timestamps
    
    def reset_memory(self):
        """Reset all memory to zero."""
        self.memory.zero_()
        self.last_update_time.zero_()


class TemporalGraphAttention(nn.Module):
    """
    Temporal Graph Attention layer with time encoding.
    
    Combines spatial graph attention with temporal information.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_dim: int = 100,
        use_edge_attr: bool = True
    ):
        """
        Initialize temporal graph attention.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            time_dim: Time encoding dimension
            use_edge_attr: Whether to use edge attributes
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.time_dim = time_dim
        self.use_edge_attr = use_edge_attr
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim + time_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        
        # Edge attribute transformation (if used)
        if use_edge_attr:
            self.W_e = nn.Linear(time_dim, out_dim)
        
        # Output projection
        self.W_o = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_encoding: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            time_encoding: Time encoding for edges [num_edges, time_dim]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # Compute queries
        Q = self.W_q(x)  # [num_nodes, out_dim]
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute keys with time encoding
        if time_encoding is not None:
            # Expand time encoding to match nodes
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # Concatenate node features with time encoding
            x_src = x[src_nodes]  # [num_edges, in_dim]
            x_with_time = torch.cat([x_src, time_encoding], dim=-1)
            
            K = self.W_k(x_with_time)  # [num_edges, out_dim]
        else:
            # No time encoding - use zero padding
            src_nodes = edge_index[0]
            x_src = x[src_nodes]  # [num_edges, in_dim]
            x_with_time = torch.cat([x_src, torch.zeros(num_edges, self.time_dim, device=x.device)], dim=-1)
            K = self.W_k(x_with_time)  # [num_edges, out_dim]
        
        K = K.view(num_edges, self.num_heads, self.head_dim)
        
        # Compute values
        V = self.W_v(x)  # [num_nodes, out_dim]
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        V = V[edge_index[0]]  # [num_edges, num_heads, head_dim]
        
        # Compute attention scores
        Q_dst = Q[edge_index[1]]  # [num_edges, num_heads, head_dim]
        attn_scores = (Q_dst * K).sum(dim=-1) * self.scale  # [num_edges, num_heads]
        
        # Add edge attributes if available
        if self.use_edge_attr and edge_attr is not None and time_encoding is not None:
            edge_features = self.W_e(time_encoding)  # [num_edges, out_dim]
            edge_features = edge_features.view(num_edges, self.num_heads, self.head_dim)
            edge_attn = (Q_dst * edge_features).sum(dim=-1) * self.scale
            attn_scores = attn_scores + edge_attn
        
        # Softmax over neighbors
        # Group by destination node
        attn_weights = torch.zeros_like(attn_scores)
        for head in range(self.num_heads):
            # Compute softmax per destination node
            for dst in range(num_nodes):
                mask = edge_index[1] == dst
                if mask.any():
                    attn_weights[mask, head] = F.softmax(attn_scores[mask, head], dim=0)
        
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate messages
        messages = attn_weights.unsqueeze(-1) * V  # [num_edges, num_heads, head_dim]
        
        # Sum messages per destination node
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, edge_index[1], messages)
        
        # Reshape and project
        out = out.view(num_nodes, self.out_dim)
        out = self.W_o(out)
        
        return out


class KairosEncoder(nn.Module):
    """
    Kairos encoder with temporal graph attention and memory.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_dim: int = 100,
        memory_dim: int = 100,
        num_nodes: int = 268435,
        use_memory: bool = True
    ):
        """
        Initialize Kairos encoder.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            time_dim: Time encoding dimension
            memory_dim: Memory dimension
            num_nodes: Maximum number of nodes
            use_memory: Whether to use temporal memory
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_memory = use_memory
        
        # Time encoder (cosine time encoding)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Temporal memory (if enabled)
        if use_memory:
            self.memory = TemporalMemory(
                num_nodes=num_nodes,
                memory_dim=memory_dim,
                time_dim=time_dim,
                message_dim=hidden_dim,
                updater_type='gru'
            )
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Temporal graph attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dim = hidden_dim
            layer_out_dim = out_dim if i == num_layers - 1 else hidden_dim
            
            self.layers.append(
                TemporalGraphAttention(
                    in_dim=layer_in_dim,
                    out_dim=layer_out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    time_dim=time_dim,
                    use_edge_attr=True
                )
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes (optional)
            timestamps: Edge timestamps [num_edges] (optional)
            
        Returns:
            Node embeddings [num_nodes, out_dim]
        """
        # Encode time if available
        time_encoding = None
        if timestamps is not None:
            # Normalize timestamps to [0, 1] range
            if timestamps.max() > 1:
                timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
            else:
                timestamps_norm = timestamps
            
            time_encoding = self.time_encoder(timestamps_norm.unsqueeze(-1))
        
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Apply temporal graph attention layers
        for layer in self.layers:
            h_new = layer(h, edge_index, time_encoding, edge_attr)
            h = h + h_new  # Residual connection
            h = F.relu(h)
            h = self.dropout(h)
        
        return h


class KairosDecoder(nn.Module):
    """
    Kairos decoder for edge-level anomaly detection.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = 2
    ):
        """
        Initialize decoder.
        
        Args:
            in_dim: Input dimension (from encoder)
            hidden_dim: Hidden dimension
            out_dim: Output dimension (2 for binary classification)
        """
        super().__init__()
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        edge_labels: Optional[torch.Tensor] = None,
        inference: bool = False
    ) -> torch.Tensor:
        """
        Predict edge labels.
        
        Args:
            h_src: Source node embeddings [num_edges, in_dim]
            h_dst: Destination node embeddings [num_edges, in_dim]
            edge_features: Edge features (optional)
            edge_labels: Ground truth labels (for training)
            inference: Whether in inference mode
            
        Returns:
            Edge predictions [num_edges, out_dim] or [num_edges] for binary
        """
        # Concatenate source and destination embeddings
        edge_emb = torch.cat([h_src, h_dst], dim=-1)
        
        # Predict
        logits = self.edge_predictor(edge_emb)
        
        if inference:
            # Return probabilities for anomaly class
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1]  # Anomaly probability
        else:
            return logits
