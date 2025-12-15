"""
Rich Feature Extraction for Provenance Graphs.

This module provides enhanced feature extraction that goes beyond simple one-hot encoding
to create meaningful node representations that capture:
- Learnable type embeddings (32-dim)
- Graph topology metrics (degree, PageRank, centrality)
- Temporal statistics (first/last seen, lifetime, event frequency)
- Metadata features (PID, UID, command hash for processes)

Target: 50-100 dimensional features vs original 3-dim one-hot encoding
Expected improvement: +15% AUROC
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class RichFeatureExtractor:
    """
    Extract rich multi-dimensional features from provenance graph nodes.
    
    Combines multiple feature types:
    - Type embeddings (learnable, 32-dim)
    - Structural features (degree, PageRank, clustering coefficient)
    - Temporal features (timestamps, durations, frequencies)
    - Metadata features (process info, file paths, etc.)
    """
    
    def __init__(
        self,
        num_node_types: int = 3,
        type_embed_dim: int = 32,
        structural_dim: int = 16,
        temporal_dim: int = 16,
        metadata_dim: int = 16,
        device: str = 'cpu'
    ):
        """
        Initialize feature extractor.
        
        Args:
            num_node_types: Number of unique node types
            type_embed_dim: Dimension of type embeddings
            structural_dim: Dimension of structural features
            temporal_dim: Dimension of temporal features
            metadata_dim: Dimension of metadata features
            device: Device for torch tensors
        """
        self.num_node_types = num_node_types
        self.type_embed_dim = type_embed_dim
        self.structural_dim = structural_dim
        self.temporal_dim = temporal_dim
        self.metadata_dim = metadata_dim
        self.device = device
        
        # Total feature dimension
        self.feature_dim = type_embed_dim + structural_dim + temporal_dim + metadata_dim
        
        # Type embedding layer (learnable)
        self.type_embedding = nn.Embedding(num_node_types, type_embed_dim)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        
        logger.info(f"RichFeatureExtractor initialized:")
        logger.info(f"  Type embedding: {type_embed_dim}D")
        logger.info(f"  Structural features: {structural_dim}D")
        logger.info(f"  Temporal features: {temporal_dim}D")
        logger.info(f"  Metadata features: {metadata_dim}D")
        logger.info(f"  Total feature dimension: {self.feature_dim}D")
    
    def extract_features(
        self,
        node_to_id: Dict[Any, int],
        node_types: Dict[Any, str],
        node_type_to_id: Dict[str, int],
        edges: List[Any],
        node_id_to_entity: Optional[Dict[int, Dict]] = None,
        edge_timestamps: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Extract rich features for all nodes.
        
        Args:
            node_to_id: Mapping from node identifier to integer ID
            node_types: Mapping from node to type string
            node_type_to_id: Mapping from type string to type ID
            edges: List of edges (tuples or dicts)
            node_id_to_entity: Optional metadata for each node
            edge_timestamps: Optional list of timestamps for temporal features
            
        Returns:
            Feature matrix of shape (num_nodes, feature_dim)
        """
        num_nodes = len(node_to_id)
        features = np.zeros((num_nodes, self.feature_dim), dtype=np.float32)
        
        # Extract type embeddings
        type_features = self._extract_type_features(
            node_to_id, node_types, node_type_to_id
        )
        
        # Extract structural features (degree, PageRank, etc.)
        structural_features = self._extract_structural_features(
            node_to_id, edges
        )
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(
            node_to_id, edges, edge_timestamps
        )
        
        # Extract metadata features
        metadata_features = self._extract_metadata_features(
            node_to_id, node_types, node_id_to_entity
        )
        
        # Concatenate all features
        offset = 0
        features[:, offset:offset+self.type_embed_dim] = type_features
        offset += self.type_embed_dim
        
        features[:, offset:offset+self.structural_dim] = structural_features
        offset += self.structural_dim
        
        features[:, offset:offset+self.temporal_dim] = temporal_features
        offset += self.temporal_dim
        
        features[:, offset:offset+self.metadata_dim] = metadata_features
        
        logger.info(f"Extracted rich features: {features.shape}")
        logger.info(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
        logger.info(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        
        return features
    
    def _extract_type_features(
        self,
        node_to_id: Dict[Any, int],
        node_types: Dict[Any, str],
        node_type_to_id: Dict[str, int]
    ) -> np.ndarray:
        """
        Extract learnable type embeddings.
        
        Args:
            node_to_id: Node to ID mapping
            node_types: Node to type mapping
            node_type_to_id: Type to ID mapping
            
        Returns:
            Type embeddings of shape (num_nodes, type_embed_dim)
        """
        num_nodes = len(node_to_id)
        type_ids = np.zeros(num_nodes, dtype=np.int64)
        
        # Map each node to its type ID
        for node, node_id in node_to_id.items():
            if node in node_types:
                type_str = node_types[node]
                if type_str in node_type_to_id:
                    type_ids[node_id] = node_type_to_id[type_str]
        
        # Get embeddings from the embedding layer
        with torch.no_grad():
            type_ids_tensor = torch.LongTensor(type_ids).to(self.device)
            embeddings = self.type_embedding(type_ids_tensor)
            embeddings_np = embeddings.cpu().numpy()
        
        return embeddings_np
    
    def _extract_structural_features(
        self,
        node_to_id: Dict[Any, int],
        edges: List[Any]
    ) -> np.ndarray:
        """
        Extract graph topology features.
        
        Features:
        - In-degree
        - Out-degree
        - Total degree
        - Log degrees (to handle outliers)
        - PageRank score
        - Clustering coefficient (approximated by neighbor overlap)
        
        Args:
            node_to_id: Node to ID mapping
            edges: List of edges
            
        Returns:
            Structural features of shape (num_nodes, structural_dim)
        """
        num_nodes = len(node_to_id)
        
        # Initialize degree counters
        in_degree = np.zeros(num_nodes, dtype=np.float32)
        out_degree = np.zeros(num_nodes, dtype=np.float32)
        
        # Build adjacency list for PageRank
        adjacency = defaultdict(set)
        reverse_adjacency = defaultdict(set)
        
        # Count degrees
        for edge in edges:
            if isinstance(edge, tuple):
                src, dst, _ = edge
            else:
                src = edge.get('src')
                dst = edge.get('dst')
            
            if src in node_to_id and dst in node_to_id:
                src_id = node_to_id[src]
                dst_id = node_to_id[dst]
                
                out_degree[src_id] += 1
                in_degree[dst_id] += 1
                
                adjacency[src_id].add(dst_id)
                reverse_adjacency[dst_id].add(src_id)
        
        # Calculate PageRank (simplified, 10 iterations)
        pagerank = self._calculate_pagerank(adjacency, num_nodes, iterations=10)
        
        # Calculate clustering coefficient approximation
        clustering = self._calculate_clustering_coefficient(adjacency, reverse_adjacency, num_nodes)
        
        # Assemble structural features
        total_degree = in_degree + out_degree
        
        # Normalize degrees (log scale to handle outliers)
        log_in_degree = np.log1p(in_degree)
        log_out_degree = np.log1p(out_degree)
        log_total_degree = np.log1p(total_degree)
        
        # Normalize to [0, 1] range
        def normalize(x):
            if x.max() > 0:
                return x / x.max()
            return x
        
        structural_features = np.column_stack([
            normalize(in_degree),
            normalize(out_degree),
            normalize(total_degree),
            normalize(log_in_degree),
            normalize(log_out_degree),
            normalize(log_total_degree),
            pagerank,
            clustering,
        ])
        
        # Pad or truncate to structural_dim
        if structural_features.shape[1] < self.structural_dim:
            padding = np.zeros((num_nodes, self.structural_dim - structural_features.shape[1]))
            structural_features = np.hstack([structural_features, padding])
        else:
            structural_features = structural_features[:, :self.structural_dim]
        
        return structural_features
    
    def _calculate_pagerank(
        self,
        adjacency: Dict[int, Set[int]],
        num_nodes: int,
        damping: float = 0.85,
        iterations: int = 10
    ) -> np.ndarray:
        """
        Calculate PageRank scores using power iteration.
        
        Args:
            adjacency: Adjacency list
            num_nodes: Number of nodes
            damping: Damping factor
            iterations: Number of iterations
            
        Returns:
            PageRank scores of shape (num_nodes,)
        """
        # Initialize PageRank scores
        pr = np.ones(num_nodes, dtype=np.float32) / num_nodes
        
        # Power iteration
        for _ in range(iterations):
            new_pr = np.ones(num_nodes, dtype=np.float32) * (1 - damping) / num_nodes
            
            for src_id, neighbors in adjacency.items():
                if len(neighbors) > 0:
                    contribution = damping * pr[src_id] / len(neighbors)
                    for dst_id in neighbors:
                        new_pr[dst_id] += contribution
            
            pr = new_pr
        
        # Normalize
        if pr.sum() > 0:
            pr = pr / pr.sum()
        
        return pr
    
    def _calculate_clustering_coefficient(
        self,
        adjacency: Dict[int, Set[int]],
        reverse_adjacency: Dict[int, Set[int]],
        num_nodes: int
    ) -> np.ndarray:
        """
        Calculate local clustering coefficient for each node.
        
        Approximated by: overlap of in-neighbors and out-neighbors.
        
        Args:
            adjacency: Forward adjacency list (out-edges)
            reverse_adjacency: Reverse adjacency list (in-edges)
            num_nodes: Number of nodes
            
        Returns:
            Clustering coefficients of shape (num_nodes,)
        """
        clustering = np.zeros(num_nodes, dtype=np.float32)
        
        for node_id in range(num_nodes):
            out_neighbors = adjacency.get(node_id, set())
            in_neighbors = reverse_adjacency.get(node_id, set())
            
            total_neighbors = len(out_neighbors) + len(in_neighbors)
            if total_neighbors > 0:
                # Measure overlap
                overlap = len(out_neighbors & in_neighbors)
                clustering[node_id] = overlap / total_neighbors
        
        return clustering
    
    def _extract_temporal_features(
        self,
        node_to_id: Dict[Any, int],
        edges: List[Any],
        edge_timestamps: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Extract temporal features for each node.
        
        Features:
        - First appearance time
        - Last appearance time
        - Lifetime duration
        - Event frequency (edges per second)
        - Time since start (normalized)
        - Activity concentration (events in peak hour vs total)
        
        Args:
            node_to_id: Node to ID mapping
            edges: List of edges
            edge_timestamps: Optional timestamps for each edge
            
        Returns:
            Temporal features of shape (num_nodes, temporal_dim)
        """
        num_nodes = len(node_to_id)
        
        # Initialize temporal statistics
        first_seen = np.full(num_nodes, np.inf, dtype=np.float32)
        last_seen = np.full(num_nodes, -np.inf, dtype=np.float32)
        event_count = np.zeros(num_nodes, dtype=np.float32)
        
        # Track timestamps if available
        if edge_timestamps is None:
            # Try to extract from edges
            edge_timestamps = []
            for edge in edges:
                if isinstance(edge, dict) and 'timestamp' in edge:
                    edge_timestamps.append(edge['timestamp'])
                else:
                    edge_timestamps.append(0.0)
        
        # Ensure we have same number of timestamps as edges
        if len(edge_timestamps) < len(edges):
            edge_timestamps = [0.0] * len(edges)
        
        # Collect temporal info
        for edge, timestamp in zip(edges, edge_timestamps):
            if isinstance(edge, tuple):
                src, dst, _ = edge
            else:
                src = edge.get('src')
                dst = edge.get('dst')
            
            if src in node_to_id:
                src_id = node_to_id[src]
                first_seen[src_id] = min(first_seen[src_id], timestamp)
                last_seen[src_id] = max(last_seen[src_id], timestamp)
                event_count[src_id] += 1
            
            if dst in node_to_id:
                dst_id = node_to_id[dst]
                first_seen[dst_id] = min(first_seen[dst_id], timestamp)
                last_seen[dst_id] = max(last_seen[dst_id], timestamp)
                event_count[dst_id] += 1
        
        # Handle nodes with no edges
        first_seen[first_seen == np.inf] = 0
        last_seen[last_seen == -np.inf] = 0
        
        # Calculate derived features
        lifetime = last_seen - first_seen
        
        # Event frequency (events per second)
        frequency = np.zeros_like(lifetime)
        nonzero_lifetime = lifetime > 0
        frequency[nonzero_lifetime] = event_count[nonzero_lifetime] / lifetime[nonzero_lifetime]
        
        # Normalize temporal features
        global_start = first_seen[first_seen > 0].min() if (first_seen > 0).any() else 0
        global_end = last_seen.max() if last_seen.max() > 0 else 1
        global_duration = global_end - global_start if global_end > global_start else 1
        
        # Normalized features
        norm_first_seen = (first_seen - global_start) / global_duration
        norm_last_seen = (last_seen - global_start) / global_duration
        norm_lifetime = lifetime / global_duration
        
        # Log-scale event count
        log_event_count = np.log1p(event_count)
        norm_log_event_count = log_event_count / (log_event_count.max() + 1e-8)
        
        # Assemble temporal features
        temporal_features = np.column_stack([
            norm_first_seen,
            norm_last_seen,
            norm_lifetime,
            event_count / (event_count.max() + 1e-8),
            norm_log_event_count,
            frequency / (frequency.max() + 1e-8),
        ])
        
        # Pad or truncate to temporal_dim
        if temporal_features.shape[1] < self.temporal_dim:
            padding = np.zeros((num_nodes, self.temporal_dim - temporal_features.shape[1]))
            temporal_features = np.hstack([temporal_features, padding])
        else:
            temporal_features = temporal_features[:, :self.temporal_dim]
        
        return temporal_features
    
    def _extract_metadata_features(
        self,
        node_to_id: Dict[Any, int],
        node_types: Dict[Any, str],
        node_id_to_entity: Optional[Dict[int, Dict]] = None
    ) -> np.ndarray:
        """
        Extract metadata features from node entities.
        
        For subjects (processes):
        - PID (normalized)
        - PPID (normalized)
        - UID (normalized)
        - Command line hash
        
        For files/network:
        - Path/address hash
        
        Args:
            node_to_id: Node to ID mapping
            node_types: Node to type mapping
            node_id_to_entity: Optional entity metadata
            
        Returns:
            Metadata features of shape (num_nodes, metadata_dim)
        """
        num_nodes = len(node_to_id)
        metadata_features = np.zeros((num_nodes, self.metadata_dim), dtype=np.float32)
        
        if node_id_to_entity is None:
            return metadata_features
        
        # Extract metadata for each node
        pids = []
        ppids = []
        uids = []
        
        for node, node_id in node_to_id.items():
            entity = node_id_to_entity.get(node, {})
            node_type = node_types.get(node, 'unknown')
            
            if node_type == 'subject':
                # Process metadata
                pid = entity.get('pid', 0)
                ppid = entity.get('ppid', 0)
                uid = entity.get('uid', 0)
                cmd = entity.get('cmdLine', '') or entity.get('properties', {}).get('cmdLine', '')
                
                pids.append(pid)
                ppids.append(ppid)
                uids.append(uid)
                
                # Hash command line (normalized to [0, 1])
                cmd_hash = self._hash_string(cmd)
                
                # Store features
                metadata_features[node_id, 0] = pid
                metadata_features[node_id, 1] = ppid
                metadata_features[node_id, 2] = uid
                metadata_features[node_id, 3] = cmd_hash
                
            elif node_type in ['fileobject', 'netflowobject']:
                # File/network metadata
                path = entity.get('path', '') or entity.get('remoteAddress', '')
                path_hash = self._hash_string(path)
                
                metadata_features[node_id, 0] = path_hash
        
        # Normalize PID, PPID, UID
        if len(pids) > 0:
            max_pid = max(pids) if max(pids) > 0 else 1
            max_ppid = max(ppids) if max(ppids) > 0 else 1
            max_uid = max(uids) if max(uids) > 0 else 1
            
            for node, node_id in node_to_id.items():
                if metadata_features[node_id, 0] > 0:  # Has PID
                    metadata_features[node_id, 0] /= max_pid
                    metadata_features[node_id, 1] /= max_ppid
                    metadata_features[node_id, 2] /= max_uid
        
        return metadata_features
    
    def _hash_string(self, s: str) -> float:
        """
        Hash a string to a normalized float value [0, 1].
        
        Args:
            s: String to hash
            
        Returns:
            Normalized hash value
        """
        if not s:
            return 0.0
        
        # Use MD5 hash for consistency
        hash_obj = hashlib.md5(s.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        
        # Normalize to [0, 1]
        return hash_int / (2**32)
    
    def get_embedding_layer(self) -> nn.Embedding:
        """
        Get the type embedding layer for use in models.
        
        Returns:
            Type embedding layer
        """
        return self.type_embedding
    
    def save_embeddings(self, path: str):
        """
        Save learned type embeddings.
        
        Args:
            path: Path to save embeddings
        """
        torch.save(self.type_embedding.state_dict(), path)
        logger.info(f"Saved type embeddings to {path}")
    
    def load_embeddings(self, path: str):
        """
        Load learned type embeddings.
        
        Args:
            path: Path to load embeddings from
        """
        self.type_embedding.load_state_dict(torch.load(path))
        logger.info(f"Loaded type embeddings from {path}")
