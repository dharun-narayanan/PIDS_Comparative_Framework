"""
Sketch Generator for ThreaTrace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SketchGenerator(nn.Module):
    """
    Generates fixed-size sketch representations from variable-size graphs
    
    Uses statistical features and learned projections to create
    compact graph representations suitable for efficient anomaly detection.
    """
    
    def __init__(self, input_dim, sketch_dim=256, use_statistics=True):
        """
        Args:
            input_dim: Input feature dimension
            sketch_dim: Output sketch dimension
            use_statistics: Whether to include statistical features
        """
        super(SketchGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        self.use_statistics = use_statistics
        
        # Learnable projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, sketch_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(sketch_dim * 2, sketch_dim)
        )
        
        # Statistical feature aggregation weights
        if use_statistics:
            self.stat_weights = nn.Parameter(torch.ones(4))  # mean, max, min, std
    
    def forward(self, x, batch=None):
        """
        Generate sketch from node features
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            sketch: Fixed-size sketch representation [batch_size, sketch_dim]
        """
        if batch is None:
            # Single graph case
            return self._sketch_single_graph(x)
        else:
            # Batch of graphs
            return self._sketch_batch(x, batch)
    
    def _sketch_single_graph(self, x):
        """Generate sketch for a single graph"""
        # Project features
        projected = self.projection(x)
        
        if self.use_statistics:
            # Compute statistical features
            mean_feat = projected.mean(dim=0, keepdim=True)
            max_feat = projected.max(dim=0, keepdim=True)[0]
            min_feat = projected.min(dim=0, keepdim=True)[0]
            std_feat = projected.std(dim=0, keepdim=True)
            
            # Weighted combination
            weights = F.softmax(self.stat_weights, dim=0)
            sketch = (weights[0] * mean_feat + 
                     weights[1] * max_feat + 
                     weights[2] * min_feat + 
                     weights[3] * std_feat)
        else:
            # Simple mean pooling
            sketch = projected.mean(dim=0, keepdim=True)
        
        return sketch
    
    def _sketch_batch(self, x, batch):
        """Generate sketches for a batch of graphs"""
        batch_size = batch.max().item() + 1
        sketches = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                graph_features = x[mask]
                sketch = self._sketch_single_graph(graph_features)
                sketches.append(sketch)
        
        return torch.cat(sketches, dim=0)
    
    def compute_sketch_distance(self, sketch1, sketch2, metric='cosine'):
        """
        Compute distance between two sketches
        
        Args:
            sketch1: First sketch [sketch_dim]
            sketch2: Second sketch [sketch_dim]
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            distance: Scalar distance value
        """
        if metric == 'cosine':
            return 1 - F.cosine_similarity(sketch1, sketch2, dim=-1)
        elif metric == 'euclidean':
            return torch.norm(sketch1 - sketch2, p=2, dim=-1)
        elif metric == 'manhattan':
            return torch.norm(sketch1 - sketch2, p=1, dim=-1)
        else:
            raise ValueError(f"Unknown metric: {metric}")


class MinHashSketch(nn.Module):
    """
    MinHash-based sketch for graph similarity
    
    Uses randomized hashing for efficient similarity estimation.
    """
    
    def __init__(self, feature_dim, num_hashes=128, seed=42):
        """
        Args:
            feature_dim: Input feature dimension
            num_hashes: Number of hash functions
            seed: Random seed for reproducibility
        """
        super(MinHashSketch, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_hashes = num_hashes
        
        # Initialize random hash matrices
        torch.manual_seed(seed)
        self.hash_matrices = nn.Parameter(
            torch.randn(num_hashes, feature_dim),
            requires_grad=False
        )
    
    def forward(self, x, batch=None):
        """
        Compute MinHash sketch
        
        Args:
            x: Node features [num_nodes, feature_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            sketch: MinHash sketch [batch_size, num_hashes]
        """
        # Compute hash values
        hash_values = torch.matmul(x, self.hash_matrices.t())  # [num_nodes, num_hashes]
        
        if batch is None:
            # Single graph: take minimum hash value for each hash function
            sketch = hash_values.min(dim=0, keepdim=True)[0]
        else:
            # Batch of graphs
            batch_size = batch.max().item() + 1
            sketch = torch.zeros(batch_size, self.num_hashes, device=x.device)
            
            for i in range(batch_size):
                mask = (batch == i)
                if mask.sum() > 0:
                    sketch[i] = hash_values[mask].min(dim=0)[0]
        
        return sketch
    
    def estimate_jaccard_similarity(self, sketch1, sketch2):
        """
        Estimate Jaccard similarity from MinHash sketches
        
        Args:
            sketch1: First sketch [num_hashes]
            sketch2: Second sketch [num_hashes]
            
        Returns:
            similarity: Estimated Jaccard similarity
        """
        matches = (sketch1 == sketch2).float().sum()
        similarity = matches / self.num_hashes
        return similarity
