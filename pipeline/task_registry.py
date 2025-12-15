"""
Registry of all available pipeline tasks.

This module contains the actual task implementations for:
1. Data loading and preprocessing
2. Graph construction
3. Feature extraction
4. Graph transformation
5. Featurization
6. Batch construction
7. Model training
8. Model inference/evaluation
9. Metrics calculation
10. Post-processing (optional attack tracing)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from utils.rich_features import RichFeatureExtractor

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry of all pipeline tasks."""
    
    @staticmethod
    def load_preprocessed_data(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 1: Load preprocessed dataset.
        
        Loads the preprocessed graph data from pickle file created by
        the preprocessing script.
        
        Args:
            config: Global configuration
            task_config: Task-specific config (data_path, dataset_name)
            dependencies: Results from dependent tasks (none for this task)
            
        Returns:
            Dictionary containing loaded graph data
        """
        data_path = Path(task_config.get('data_path', config['data']['path']))
        dataset_name = task_config.get('dataset_name', config['data']['dataset'])
        
        logger.info(f"Loading preprocessed data from {data_path}")
        
        # Check if data_path is a direct .pkl file or a directory
        if data_path.is_file() and data_path.suffix == '.pkl':
            # Direct file path
            pkl_file = data_path
            logger.info(f"Loading from {pkl_file}")
        else:
            # Directory path - look for preprocessed pickle file matching dataset name
            # First try to find file matching the dataset name exactly
            dataset_pkl = data_path / f"{dataset_name}_graph.pkl"
            if dataset_pkl.exists():
                pkl_file = dataset_pkl
                logger.info(f"Loading from {pkl_file}")
            else:
                # Fallback: look for any .pkl file
                pkl_files = list(data_path.glob('*.pkl'))
                if not pkl_files:
                    raise FileNotFoundError(f"No .pkl files found in {data_path}")
                
                # Load the first pkl file (assuming single dataset)
                pkl_file = pkl_files[0]
                logger.info(f"Loading from {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Log statistics
        if isinstance(graph_data, dict):
            num_nodes = graph_data.get('num_nodes', 0)
            num_edges = len(graph_data.get('edges', graph_data.get('events', [])))
            logger.info(f"Loaded graph: {num_nodes} nodes, {num_edges} edges")
            if 'stats' in graph_data:
                logger.info(f"Statistics: {graph_data['stats']}")
        
        return {
            'graph_data': graph_data,
            'data_path': data_path,
            'dataset_name': dataset_name
        }
    
    @staticmethod
    def construct_time_windows(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 2: Construct time-window based graphs.
        
        Takes the loaded graph data and constructs temporal windows
        for streaming graph processing. This matches the paper methodology
        where graphs are divided into hourly/configurable windows.
        
        Benefits of time windowing:
        - Models learn temporal anomalies (unusual for THIS time period)
        - Reduces memory footprint (smaller graphs)
        - Enables online/streaming processing
        - Matches MAGIC/Kairos paper evaluation
        
        Args:
            config: Global configuration
            task_config: Task config (window_size, overlap, min_edges, etc.)
            dependencies: Must contain 'load_preprocessed_data' result
            
        Returns:
            List of time-windowed graphs
        """
        graph_data = dependencies['load_preprocessed_data']['graph_data']
        
        window_size = task_config.get('window_size', 3600)  # seconds (default 1 hour)
        overlap = task_config.get('overlap', 0.0)  # fraction (0 = no overlap)
        min_edges_per_window = task_config.get('min_edges', 10)  # Skip windows with too few edges
        max_windows = task_config.get('max_windows', None)  # Limit total windows (for memory)
        
        logger.info(f"Constructing time windows:")
        logger.info(f"  Window size: {window_size}s ({window_size/3600:.2f} hours)")
        logger.info(f"  Overlap: {overlap*100:.0f}%")
        logger.info(f"  Min edges per window: {min_edges_per_window}")
        
        edges = graph_data.get('edges', graph_data.get('events', []))
        
        # Extract timestamps from edges
        edges_with_timestamps = []
        for edge in edges:
            if isinstance(edge, dict):
                timestamp = edge.get('timestamp', 0)
                edges_with_timestamps.append((timestamp, edge))
            elif isinstance(edge, tuple) and len(edge) >= 3:
                # Tuple format - try to find timestamp in graph_data
                # For now, assume no timestamp (will be handled below)
                edges_with_timestamps.append((0, edge))
        
        # Sort by timestamp
        edges_with_timestamps.sort(key=lambda x: x[0])
        
        # Create time windows
        time_windows = []
        
        if edges_with_timestamps:
            timestamps = [t for t, _ in edges_with_timestamps]
            min_time = min(timestamps)
            max_time = max(timestamps)
            
            # Handle case where all timestamps are 0 (no temporal info)
            if max_time == min_time:
                logger.warning("No temporal information found in edges - creating single window")
                time_windows.append({
                    'window_id': 0,
                    'start_time': 0,
                    'end_time': float('inf'),
                    'edges': [edge for _, edge in edges_with_timestamps],
                    'num_edges': len(edges_with_timestamps),
                    'has_temporal_info': False
                })
            else:
                # Create windows with overlap
                current_start = min_time
                stride = window_size * (1 - overlap)
                window_id = 0
                
                logger.info(f"  Time range: {min_time:.0f} to {max_time:.0f} ({(max_time-min_time)/3600:.2f} hours)")
                
                while current_start < max_time:
                    current_end = current_start + window_size
                    
                    # Get edges in this window
                    window_edges = [
                        edge for timestamp, edge in edges_with_timestamps
                        if current_start <= timestamp < current_end
                    ]
                    
                    # Only create window if it has enough edges
                    if len(window_edges) >= min_edges_per_window:
                        # Calculate window statistics
                        window_timestamps = [t for t, e in edges_with_timestamps if current_start <= t < current_end]
                        
                        time_windows.append({
                            'window_id': window_id,
                            'start_time': current_start,
                            'end_time': current_end,
                            'edges': window_edges,
                            'num_edges': len(window_edges),
                            'has_temporal_info': True,
                            'duration': current_end - current_start,
                            'edge_rate': len(window_edges) / (current_end - current_start) if current_end > current_start else 0
                        })
                        window_id += 1
                        
                        # Limit total windows if specified
                        if max_windows and len(time_windows) >= max_windows:
                            logger.info(f"  Reached max_windows limit ({max_windows}), stopping")
                            break
                    
                    current_start += stride
        else:
            logger.warning("No edges found in graph data")
        
        logger.info(f"✓ Created {len(time_windows)} time windows")
        
        # Log window statistics
        if time_windows:
            window_sizes = [w['num_edges'] for w in time_windows]
            logger.info(f"  Window sizes: min={min(window_sizes)}, max={max(window_sizes)}, mean={np.mean(window_sizes):.0f}")
            
            # Show first few windows
            for i, window in enumerate(time_windows[:3]):
                logger.info(f"  Window {i}: {window['num_edges']} edges, "
                          f"time=[{window['start_time']:.0f}, {window['end_time']:.0f}]")
        
        return {
            'time_windows': time_windows,
            'window_size': window_size,
            'overlap': overlap,
            'original_graph': graph_data,
            'num_windows': len(time_windows),
            'windowing_enabled': len(time_windows) > 1
        }
    
    @staticmethod
    def graph_transformation(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 3: Apply graph transformations.
        
        Applies transformations like:
        - Undirected conversion
        - DAG creation
        - Edge deduplication
        - Graph simplification
        
        Args:
            config: Global configuration
            task_config: Transformation config
            dependencies: Must contain time window data
            
        Returns:
            Transformed graph data
        """
        # Get time windows and original graph data
        if 'construct_time_windows' in dependencies:
            time_windows = dependencies['construct_time_windows']['time_windows']
            graph_data = dependencies['construct_time_windows']['original_graph']
        else:
            # Fallback: work with original graph
            loaded_data = dependencies['load_preprocessed_data']['graph_data']
            
            # Handle case where graph_data is a list of graphs (from preprocessing)
            if isinstance(loaded_data, list):
                if len(loaded_data) == 0:
                    raise ValueError("Loaded graph data is an empty list")
                # Take the first graph if it's a list
                graph_data = loaded_data[0]
                logger.info(f"Extracted first graph from list of {len(loaded_data)} graphs")
            else:
                graph_data = loaded_data
            
            # Ensure graph_data is a dictionary
            if not isinstance(graph_data, dict):
                raise TypeError(f"Expected graph_data to be dict, got {type(graph_data)}")
            
            time_windows = [{'edges': graph_data.get('edges', []), 'original': True}]
        
        # Extract labels if available
        labels = graph_data.get('labels', None) if not isinstance(graph_data, list) else None
        
        transform_type = task_config.get('type', 'none')
        
        logger.info(f"Applying graph transformation: {transform_type}")
        
        transformed_windows = []
        for window in time_windows:
            edges = window['edges']
            
            # Convert edges to dict format with labels if labels are available
            if labels is not None and len(labels) == len(edges):
                edges_with_labels = []
                for i, edge in enumerate(edges):
                    if isinstance(edge, tuple):
                        # Convert tuple to dict and add label
                        src, dst, edge_type_id = edge
                        edge_dict = {
                            'src': src,
                            'dst': dst,
                            'type_id': edge_type_id,
                            'label': int(labels[i])
                        }
                        edges_with_labels.append(edge_dict)
                    else:
                        # Already dict, just add label if not present
                        edge_copy = edge.copy()
                        if 'label' not in edge_copy:
                            edge_copy['label'] = int(labels[i])
                        edges_with_labels.append(edge_copy)
                edges = edges_with_labels
            
            if transform_type == 'undirected':
                # Make graph undirected (add reverse edges)
                new_edges = []
                for edge in edges:
                    new_edges.append(edge)
                    # Add reverse edge
                    if isinstance(edge, tuple):
                        # Tuple format: (src, dst, edge_type_id)
                        src, dst, edge_type = edge
                        reverse_edge = (dst, src, edge_type)
                    else:
                        # Dictionary format
                        reverse_edge = edge.copy()
                        reverse_edge['src'], reverse_edge['dst'] = edge.get('dst'), edge.get('src')
                    new_edges.append(reverse_edge)
                edges = new_edges
            
            elif transform_type == 'deduplicate':
                # Remove duplicate edges
                seen = set()
                unique_edges = []
                for edge in edges:
                    if isinstance(edge, tuple):
                        # Tuple format is already hashable
                        if edge not in seen:
                            seen.add(edge)
                            unique_edges.append(edge)
                    else:
                        # Dictionary format
                        key = (edge.get('src'), edge.get('dst'), edge.get('type'))
                        if key not in seen:
                            seen.add(key)
                            unique_edges.append(edge)
                edges = unique_edges
            
            elif transform_type == 'none':
                pass  # No transformation
            
            window_copy = window.copy()
            window_copy['edges'] = edges
            window_copy['num_edges'] = len(edges)
            transformed_windows.append(window_copy)
        
        logger.info(f"Transformed {len(transformed_windows)} graph windows")
        
        return {
            'transformed_windows': transformed_windows,
            'transform_type': transform_type,
            'graph_data': graph_data  # Pass through graph metadata
        }
    
    @staticmethod
    def feature_extraction(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 4: Extract node and edge features.
        
        Creates feature vectors for nodes and edges using various methods:
        - One-hot encoding
        - Text embeddings (Word2Vec, Doc2Vec, FastText)
        - Graph embeddings
        - Statistical features
        
        Args:
            config: Global configuration
            task_config: Feature extraction config
            dependencies: Graph data from previous tasks
            
        Returns:
            Feature matrices for nodes and edges
        """
        # Get graph data and windows
        if 'graph_transformation' in dependencies:
            windows = dependencies['graph_transformation']['transformed_windows']
            graph_data = dependencies['graph_transformation']['graph_data']
        elif 'construct_time_windows' in dependencies:
            windows = dependencies['construct_time_windows']['time_windows']
            graph_data = dependencies['construct_time_windows']['original_graph']
        else:
            graph_data = dependencies['load_preprocessed_data']['graph_data']
            windows = [{'edges': graph_data.get('edges', [])}]
        
        # Get metadata for tuple-format edges
        node_type_map = graph_data.get('node_type_map', {})
        edge_type_map = graph_data.get('edge_type_map', {})
        node_id_to_entity = graph_data.get('node_id_to_entity', {})
        
        # Create reverse mapping for edge types (id -> name)
        edge_id_to_type = {v: k for k, v in edge_type_map.items()} if edge_type_map else {}
        
        method = task_config.get('method', 'one_hot')
        node_feat_dim = task_config.get('node_feat_dim', 128)
        edge_feat_dim = task_config.get('edge_feat_dim', 64)
        
        logger.info(f"Extracting features using method: {method}")
        logger.info(f"Node type map contains {len(node_type_map)} entries")
        logger.info(f"Edge type map contains {len(edge_type_map)} entries")
        
        # Collect all unique nodes and edges
        all_nodes = set()
        all_edge_types = set()
        node_types = {}
        
        # First, populate node_types from the preprocessed node_type_map
        # node_type_map is {node_id: type_string}
        for node_id, node_type in node_type_map.items():
            node_types[node_id] = node_type
        
        for window in windows:
            for edge in window.get('edges', []):
                # Handle both tuple and dict formats
                if isinstance(edge, tuple):
                    # Tuple format: (src_id, dst_id, edge_type_id)
                    src, dst, edge_type_id = edge
                    all_nodes.add(src)
                    all_nodes.add(dst)
                    
                    # Node types already populated from node_type_map
                    # But ensure they're in the set
                    if src in node_type_map and src not in node_types:
                        node_types[src] = node_type_map[src]
                    if dst in node_type_map and dst not in node_types:
                        node_types[dst] = node_type_map[dst]
                    
                    # Add edge type - use id directly if no mapping available
                    if edge_type_id in edge_id_to_type:
                        all_edge_types.add(edge_id_to_type[edge_type_id])
                    else:
                        all_edge_types.add(str(edge_type_id))  # Use ID as string
                elif isinstance(edge, dict):
                    # Dictionary format
                    src = edge.get('src')
                    dst = edge.get('dst')
                    if src:
                        all_nodes.add(src)
                        # Try multiple ways to get node type
                        if 'src_type' in edge:
                            node_types[src] = edge['src_type']
                        elif src in node_type_map:
                            node_types[src] = node_type_map[src]
                    if dst:
                        all_nodes.add(dst)
                        if 'dst_type' in edge:
                            node_types[dst] = edge['dst_type']
                        elif dst in node_type_map:
                            node_types[dst] = node_type_map[dst]
                    
                    if 'type' in edge:
                        all_edge_types.add(edge['type'])
                    elif 'type_id' in edge:
                        type_id = edge['type_id']
                        if type_id in edge_id_to_type:
                            all_edge_types.add(edge_id_to_type[type_id])
                        else:
                            all_edge_types.add(str(type_id))
        
        num_nodes = len(all_nodes)
        num_edge_types = len(all_edge_types)
        
        logger.info(f"Found {num_nodes} unique nodes, {num_edge_types} edge types")
        logger.info(f"Found {len(node_types)} nodes with type information")
        
        # Log node type distribution
        if node_types:
            type_counts = {}
            for node_type in node_types.values():
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            logger.info(f"Node type distribution: {type_counts}")
        
        # Create node ID mapping
        node_to_id = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        edge_type_to_id = {et: idx for idx, et in enumerate(sorted(all_edge_types))}
        
        # Initialize feature matrices
        if method == 'one_hot':
            # One-hot encoding based on node types
            unique_node_types = set(node_types.values()) if node_types else set()
            
            # If no node types found, use configured dimension
            if len(unique_node_types) == 0:
                logger.warning(f"No node types found, using random features with dimension {node_feat_dim}")
                node_features = np.random.randn(num_nodes, node_feat_dim)
            else:
                node_type_to_id = {nt: idx for idx, nt in enumerate(sorted(unique_node_types))}
                node_features = np.zeros((num_nodes, len(unique_node_types)))
                for node, node_id in node_to_id.items():
                    if node in node_types:
                        type_id = node_type_to_id[node_types[node]]
                        node_features[node_id, type_id] = 1.0
            
            # If no edge types found, use configured dimension
            if num_edge_types == 0:
                logger.warning(f"No edge types found, using random features with dimension {edge_feat_dim}")
                edge_features = np.random.randn(1, edge_feat_dim)  # At least 1 row for indexing
            else:
                edge_features = np.eye(num_edge_types)  # One-hot for edge types
        
        elif method == 'rich':
            # Rich feature extraction (type embeddings + structural + temporal + metadata)
            # This method provides 50-100 dimensional features instead of 3-dim one-hot
            unique_node_types = set(node_types.values()) if node_types else set(['unknown'])
            node_type_to_id = {nt: idx for idx, nt in enumerate(sorted(unique_node_types))}
            
            logger.info("Using rich feature extraction (type embeddings + graph topology + temporal + metadata)")
            
            # Initialize rich feature extractor
            rich_extractor = RichFeatureExtractor(
                num_node_types=len(unique_node_types),
                type_embed_dim=32,  # Learnable type embeddings
                structural_dim=16,  # Degree, PageRank, clustering
                temporal_dim=16,    # First/last seen, lifetime, frequency
                metadata_dim=16,    # PID, UID, command hash
                device=str(task_config.get('device', 'cpu'))
            )
            
            # Extract timestamps from edges if available
            edge_timestamps = []
            for edge in windows[0].get('edges', []):
                if isinstance(edge, dict) and 'timestamp' in edge:
                    edge_timestamps.append(edge['timestamp'])
            
            # Extract rich features
            node_features = rich_extractor.extract_features(
                node_to_id=node_to_id,
                node_types=node_types,
                node_type_to_id=node_type_to_id,
                edges=windows[0].get('edges', []),
                node_id_to_entity=node_id_to_entity,
                edge_timestamps=edge_timestamps if edge_timestamps else None
            )
            
            # Save the embedding layer for later use in model
            graph_data['type_embedding'] = rich_extractor.get_embedding_layer()
            
            # Edge features (one-hot or random)
            if num_edge_types == 0:
                logger.warning(f"No edge types found, using random features with dimension {edge_feat_dim}")
                edge_features = np.random.randn(1, edge_feat_dim)
            else:
                edge_features = np.eye(num_edge_types)  # One-hot for edge types
        
        elif method == 'random':
            # Random features (baseline)
            node_features = np.random.randn(num_nodes, node_feat_dim)
            # Ensure at least 1 edge type for indexing
            edge_features = np.random.randn(max(num_edge_types, 1), edge_feat_dim)
        
        elif method == 'pretrained':
            # Load pretrained embeddings if available
            embed_path = task_config.get('embedding_path')
            if embed_path and Path(embed_path).exists():
                with open(embed_path, 'rb') as f:
                    embeddings = pickle.load(f)
                node_features = embeddings.get('node_features', np.random.randn(num_nodes, node_feat_dim))
                edge_features = embeddings.get('edge_features', np.random.randn(max(num_edge_types, 1), edge_feat_dim))
            else:
                logger.warning("Pretrained embeddings not found, using random")
                node_features = np.random.randn(num_nodes, node_feat_dim)
                edge_features = np.random.randn(max(num_edge_types, 1), edge_feat_dim)
        
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
        
        logger.info(f"Node features shape: {node_features.shape}")
        logger.info(f"Edge features shape: {edge_features.shape}")
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'node_to_id': node_to_id,
            'edge_type_to_id': edge_type_to_id,
            'num_nodes': num_nodes,
            'num_edge_types': num_edge_types,
            'windows': windows,  # Pass through windows for downstream tasks
            'graph_data': graph_data  # Pass through graph metadata for downstream tasks
        }
    
    @staticmethod
    def featurization_inference(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 5: Apply featurization model to extract embeddings.
        
        If a featurization model was trained (Word2Vec, etc.),
        apply it to get final node/edge embeddings.
        
        Args:
            config: Global configuration
            task_config: Featurization config
            dependencies: Feature extraction results
            
        Returns:
            Final embeddings
        """
        features = dependencies['feature_extraction']
        
        # For now, pass through features as-is
        # In a full implementation, this would apply trained embedding models
        logger.info("Featurization inference (pass-through for pretrained models)")
        
        return {
            'node_embeddings': features['node_features'],
            'edge_embeddings': features['edge_features'],
            'node_to_id': features['node_to_id'],
            'edge_type_to_id': features['edge_type_to_id'],
            'windows': features.get('windows', []),  # Pass through windows
            'graph_data': features.get('graph_data', {})  # Pass through graph metadata
        }
    
    @staticmethod
    def batch_construction(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 6: Construct batches for GNN training/inference.
        
        Creates PyTorch Geometric data objects with:
        - Node features
        - Edge indices
        - Edge features
        - Labels
        - Batch information
        
        Supports mini-batch sampling for large graphs to avoid OOM errors.
        
        Args:
            config: Global configuration
            task_config: Batch construction config
                - batch_size: Number of windows per batch
                - edge_batch_size: Max edges per batch (for large graph sampling)
                - sample_strategy: 'full' or 'random_edges'
            dependencies: Graph data and features
            
        Returns:
            List of batched graph data objects
        """
        from torch_geometric.data import Data, Batch
        
        # Get features
        if 'featurization_inference' in dependencies:
            feat_data = dependencies['featurization_inference']
        else:
            feat_data = dependencies['feature_extraction']
        
        # Get windows from feature data (passed through the pipeline)
        windows = feat_data.get('windows', [])
        if not windows:
            # Fallback if windows not available
            logger.warning("No windows found in feature data, creating single window")
            windows = [{'edges': []}]
        
        node_features = torch.FloatTensor(feat_data['node_embeddings'])
        edge_features = torch.FloatTensor(feat_data['edge_embeddings'])
        node_to_id = feat_data['node_to_id']
        edge_type_to_id = feat_data['edge_type_to_id']
        
        batch_size = task_config.get('batch_size', 1)
        edge_batch_size = task_config.get('edge_batch_size', None)  # None = no limit
        sample_strategy = task_config.get('sample_strategy', 'full')
        
        logger.info(f"Constructing batches (batch_size={batch_size}, edge_batch_size={edge_batch_size})")
        
        # Get metadata for tuple-format edges from feature data (passed through the pipeline)
        graph_data = feat_data.get('graph_data', {})
        edge_type_map = graph_data.get('edge_type_map', {})
        edge_id_to_type = {v: k for k, v in edge_type_map.items()} if edge_type_map else {}
        
        # Helper function to create Data object from edge list
        def create_data_object(edges_subset):
            """Create PyG Data object from edge list."""
            edge_index = []
            edge_attrs = []
            edge_labels = []
            
            for edge in edges_subset:
                # Handle both tuple and dict formats
                if isinstance(edge, tuple):
                    # Tuple format: (src_id, dst_id, edge_type_id)
                    src, dst, edge_type_id = edge
                    edge_type = edge_id_to_type.get(edge_type_id, 'unknown')
                    label = 0  # Default label for tuple format
                elif isinstance(edge, dict):
                    # Dictionary format
                    src = edge.get('src')
                    dst = edge.get('dst')
                    edge_type_id = edge.get('type_id')
                    edge_type = edge.get('type', edge_id_to_type.get(edge_type_id, 'unknown'))
                    label = edge.get('label', 0)
                else:
                    continue
                
                if src in node_to_id and dst in node_to_id:
                    src_id = node_to_id[src]
                    dst_id = node_to_id[dst]
                    edge_index.append([src_id, dst_id])
                    
                    # Get edge feature
                    if edge_type in edge_type_to_id:
                        type_id = edge_type_to_id[edge_type]
                        edge_attrs.append(edge_features[type_id])
                    else:
                        edge_attrs.append(torch.zeros(edge_features.shape[1]))
                    
                    edge_labels.append(label)
            
            if edge_index:
                return Data(
                    x=node_features,
                    edge_index=torch.LongTensor(edge_index).t().contiguous(),
                    edge_attr=torch.stack(edge_attrs) if edge_attrs else None,
                    y=torch.LongTensor(edge_labels),
                    num_nodes=len(node_to_id)
                )
            return None
        
        # Create Data objects for each window (with mini-batch support)
        data_list = []
        for i, window in enumerate(windows):
            edges = window.get('edges', [])
            
            if not edges:
                continue
            
            # If edge_batch_size is set and edges exceed limit, split into chunks
            if edge_batch_size and len(edges) > edge_batch_size:
                logger.info(f"Window {i} has {len(edges)} edges, splitting into chunks of {edge_batch_size}")
                
                # Split edges into chunks
                num_chunks = (len(edges) + edge_batch_size - 1) // edge_batch_size
                
                if sample_strategy == 'random_edges':
                    # Random sampling without replacement
                    import numpy as np
                    indices = np.random.permutation(len(edges))[:edge_batch_size]
                    edges_chunk = [edges[idx] for idx in indices]
                    data = create_data_object(edges_chunk)
                    if data:
                        data_list.append(data)
                    logger.info(f"Randomly sampled {edge_batch_size} edges from {len(edges)} total")
                else:
                    # Full coverage - split into sequential chunks
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * edge_batch_size
                        end_idx = min(start_idx + edge_batch_size, len(edges))
                        edges_chunk = edges[start_idx:end_idx]
                        
                        data = create_data_object(edges_chunk)
                        if data:
                            data_list.append(data)
                        
                        if (chunk_idx + 1) % 10 == 0:
                            logger.info(f"Processed {chunk_idx + 1}/{num_chunks} chunks for window {i}")
                    
                    logger.info(f"Split window {i} into {num_chunks} chunks")
            else:
                # Small enough to process in one go
                data = create_data_object(edges)
                if data:
                    data_list.append(data)
        
        logger.info(f"Created {len(data_list)} graph data objects")
        
        # Create batches
        batches = []
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i+batch_size]
            if len(batch_data) == 1:
                batches.append(batch_data[0])
            else:
                batches.append(Batch.from_data_list(batch_data))
        
        logger.info(f"Created {len(batches)} batches")
        
        return {
            'batches': batches,
            'data_list': data_list,
            'num_batches': len(batches)
        }
    
    @staticmethod
    def model_inference(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 7: Run model inference using pretrained weights.
        
        Loads the model using ModelBuilder, applies pretrained weights, and runs inference
        on the prepared graph batches.
        
        Args:
            config: Global configuration
            task_config: Model config (model_name, checkpoint_config, model_builder)
            dependencies: Batch construction results
            
        Returns:
            Model predictions and embeddings
        """
        model_name = task_config.get('model_name', config.get('model', 'magic'))
        device = config.get('device', 'cpu')
        dataset_name = task_config.get('dataset_name', 'custom_soc')
        
        logger.info(f"Running inference with model: {model_name}")
        
        # Get model builder from task config
        model_builder = task_config.get('model_builder')
        if model_builder is None:
            from models.model_builder import ModelBuilder
            model_builder = ModelBuilder(config_dir="configs/models")
        
        # Get checkpoint config
        checkpoint_config = task_config.get('checkpoint_config', {})
        checkpoint_enabled = checkpoint_config.get('enabled', True)
        
        # Build checkpoint path
        checkpoint_path = None
        if checkpoint_enabled:
            checkpoint_template = checkpoint_config.get('path', 'checkpoints/{model_name}/{dataset}.pt')
            
            # For unsupervised models, we want to test ALL available pretrained weights
            # not just dataset-specific ones
            checkpoint_dir = Path('checkpoints') / model_name
            all_checkpoints = []
            
            if checkpoint_dir.exists():
                # Get ALL .pt and .pkl files in the checkpoint directory
                all_checkpoints = sorted(checkpoint_dir.glob('*.pt')) + sorted(checkpoint_dir.glob('*.pkl'))
            
            if all_checkpoints:
                # Use the first available checkpoint
                # In future versions, this could iterate through all checkpoints
                checkpoint_path = str(all_checkpoints[0])
                logger.info(f"Using pretrained checkpoint: {checkpoint_path}")
                
                if len(all_checkpoints) > 1:
                    logger.info(f"Found {len(all_checkpoints)} pretrained weights for {model_name}")
                    logger.info(f"Additional checkpoints available: {[c.name for c in all_checkpoints[1:4]]}")
            else:
                checkpoint_path = None
                logger.warning(f"No pretrained weights found for {model_name} in {checkpoint_dir}")

        
        # Get input dimension from the batches
        batches = dependencies['batch_construction']['batches']
        input_dim = None
        if batches and len(batches) > 0:
            first_batch = batches[0]
            if hasattr(first_batch, 'x') and first_batch.x is not None:
                input_dim = first_batch.x.shape[1]
                logger.info(f"Detected input dimension: {input_dim}")
        
        # Build and load model
        logger.info(f"Building model: {model_name}")
        
        # Get model config overrides from task_config (from pipeline config)
        model_config_override = task_config.get('model_config_override')
        
        model = model_builder.build_and_load(
            model_name,
            dataset_name=dataset_name,
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=model_config_override,
            input_dim=input_dim
        )
        
        if checkpoint_path is None:
            logger.warning(f"Running inference with untrained model (no checkpoint found)")
        
        model.eval()
        
        # Get batches
        batches = dependencies['batch_construction']['batches']
        
        # Run inference
        all_predictions = []
        all_labels = []
        all_scores = []
        
        # Check if model is an autoencoder (reconstruction-based)
        model_config = task_config.get('model_config', {})
        model_type = model_config.get('model_type', 'autoencoder')
        is_autoencoder = model_type in ['autoencoder', 'vae', 'masked_autoencoder']
        
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch, inference=True)
                
                # Compute anomaly scores based on model type
                if is_autoencoder:
                    # For autoencoders, anomaly score = reconstruction error
                    # Higher reconstruction error = more anomalous
                    if isinstance(output, dict):
                        # Multi-decoder output
                        reconstructed = output.get('reconstruction', output.get('node_recon', list(output.values())[0]))
                    else:
                        reconstructed = output
                    
                    # Compute reconstruction error at node level
                    if hasattr(batch, 'x') and batch.x is not None:
                        # Use projected features if available (when input_projection was applied)
                        # Otherwise use original features
                        target_features = batch.x_projected if hasattr(batch, 'x_projected') else batch.x
                        
                        # Mean squared error between target and reconstructed features
                        node_recon_error = torch.mean((target_features - reconstructed) ** 2, dim=1)
                        
                        # Map node errors to edge errors (average of src and dst node errors)
                        if hasattr(batch, 'edge_index'):
                            src_nodes = batch.edge_index[0]
                            dst_nodes = batch.edge_index[1]
                            src_errors = node_recon_error[src_nodes]
                            dst_errors = node_recon_error[dst_nodes]
                            scores = (src_errors + dst_errors) / 2.0
                        else:
                            # No edges, use node errors directly
                            scores = node_recon_error
                    else:
                        # Fallback: use output magnitude as score
                        scores = torch.mean(torch.abs(reconstructed), dim=1)
                    
                    # Predictions: threshold at median
                    threshold = torch.median(scores)
                    predictions = (scores > threshold).long()
                
                else:
                    # For classifiers, use direct output
                    # Handle different output formats
                    if isinstance(output, dict):
                        # Multi-decoder output, use primary decoder
                        primary_key = list(output.keys())[0]
                        output = output[primary_key]
                    
                    if isinstance(output, torch.Tensor):
                        # Check if output is node-level and we need edge-level predictions
                        num_edges = batch.edge_index.shape[1] if hasattr(batch, 'edge_index') else 0
                        output_size = output.shape[0]
                        
                        # If output is node-level but we have edge labels, convert to edge-level
                        if num_edges > 0 and output_size == batch.num_nodes and hasattr(batch, 'y') and batch.y.shape[0] == num_edges:
                            # Node-level output, convert to edge-level
                            # Use reconstruction error or embedding similarity for edges
                            src_nodes = batch.edge_index[0]  # Source nodes
                            dst_nodes = batch.edge_index[1]  # Destination nodes
                            
                            if output.dim() == 2:
                                # output is [num_nodes, feature_dim]
                                # Compute edge scores as similarity between src and dst node embeddings
                                src_emb = output[src_nodes]  # [num_edges, feature_dim]
                                dst_emb = output[dst_nodes]  # [num_edges, feature_dim]
                                
                                # Cosine similarity for anomaly scores (higher similarity = lower anomaly)
                                scores = torch.nn.functional.cosine_similarity(src_emb, dst_emb, dim=1)
                                # Invert: higher score = more anomalous
                                scores = 1.0 - scores
                                predictions = (scores > 0.5).long()
                            else:
                                # Fallback: use node predictions for edges (average src and dst)
                                src_pred = output[src_nodes].squeeze()
                                dst_pred = output[dst_nodes].squeeze()
                                scores = (src_pred + dst_pred) / 2.0
                                predictions = (scores > 0.5).long()
                        elif output.dim() == 2 and output.shape[1] > 1 and output.shape[0] == num_edges:
                            # Multi-class classification (already softmax from decoder)
                            scores = output
                            predictions = torch.argmax(scores, dim=1)
                        elif output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
                            # Binary classification (already sigmoid from decoder)
                            scores = output.squeeze()
                            predictions = (scores > 0.5).long()
                        else:
                            # Multi-class or other format
                            if output.shape[0] == num_edges:
                                # Assume edge-level output
                                if output.dim() == 2 and output.shape[1] > 1:
                                    scores = output
                                    predictions = torch.argmax(scores, dim=1)
                                else:
                                    scores = output.squeeze()
                                    predictions = (scores > 0.5).long()
                            else:
                                # Fallback: create zero predictions
                                scores = torch.zeros(num_edges, device=device)
                                predictions = torch.zeros(num_edges, dtype=torch.long, device=device)
                    else:
                        scores = torch.zeros(1, device=device)
                        predictions = torch.zeros(1, dtype=torch.long, device=device)
                
                all_predictions.append(predictions.cpu())
                all_scores.append(scores.cpu())
                
                if hasattr(batch, 'y'):
                    all_labels.append(batch.y.cpu())
        
        # Concatenate results
        predictions = torch.cat(all_predictions) if all_predictions else torch.tensor([])
        scores = torch.cat(all_scores) if all_scores else torch.tensor([])
        labels = torch.cat(all_labels) if all_labels else None
        
        logger.info(f"Inference complete: {len(predictions)} predictions")
        
        return {
            'predictions': predictions.numpy(),
            'scores': scores.numpy(),
            'labels': labels.numpy() if labels is not None else None,
            'model_name': model_name
        }
    
    @staticmethod
    def calculate_metrics(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 8: Calculate evaluation metrics for unsupervised anomaly detection.
        
        Calculates both edge-level and entity-level metrics:
        - Edge-level: Traditional per-edge anomaly detection
        - Entity-level: Aggregate scores per source entity (matches paper methodology)
        
        Entity-level aggregation provides:
        - Better comparison with paper results
        - More robust anomaly detection
        - Matches MAGIC/Kairos evaluation approach
        
        Args:
            config: Global configuration
            task_config: Metrics config
            dependencies: Model inference results
            
        Returns:
            Dictionary of computed metrics (edge-level and entity-level)
        """
        inference_results = dependencies['model_inference']
        
        predictions = inference_results['predictions']
        scores = inference_results['scores']
        labels = inference_results['labels']
        
        logger.info("Calculating evaluation metrics (edge-level + entity-level aggregation)")
        
        metrics = {}
        scores_array = np.array(scores)
        
        # ========== EDGE-LEVEL METRICS (Original) ==========
        
        # Core anomaly detection metrics (always calculated)
        metrics['edge_level'] = {}
        metrics['edge_level']['anomaly_score_stats'] = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        # Percentile thresholds for anomaly detection
        metrics['edge_level']['percentiles'] = {
            '90': float(np.percentile(scores_array, 90)),
            '95': float(np.percentile(scores_array, 95)),
            '99': float(np.percentile(scores_array, 99)),
            '99.5': float(np.percentile(scores_array, 99.5)),
            '99.9': float(np.percentile(scores_array, 99.9)),
        }
        
        # Score separation metric (higher = better anomaly detection)
        if metrics['edge_level']['anomaly_score_stats']['mean'] > 0:
            metrics['edge_level']['score_separation_ratio'] = (
                metrics['edge_level']['anomaly_score_stats']['std'] / 
                metrics['edge_level']['anomaly_score_stats']['mean']
            )
        else:
            metrics['edge_level']['score_separation_ratio'] = 0.0
        
        # Count anomalies at different thresholds
        metrics['edge_level']['anomaly_counts'] = {
            'critical_99.9': int(np.sum(scores_array >= metrics['edge_level']['percentiles']['99.9'])),
            'high_99': int(np.sum(scores_array >= metrics['edge_level']['percentiles']['99'])),
            'medium_95': int(np.sum(scores_array >= metrics['edge_level']['percentiles']['95'])),
            'elevated_90': int(np.sum(scores_array >= metrics['edge_level']['percentiles']['90'])),
        }
        
        # Optional: Supervised metrics if labels are available (EDGE-LEVEL)
        # Skip for custom datasets without ground truth
        if labels is not None and len(np.unique(labels)) > 1:
            logger.info("Ground truth labels detected - calculating supervised metrics")
            try:
                from sklearn.metrics import (
                    roc_auc_score, average_precision_score,
                    precision_recall_fscore_support, confusion_matrix
                )
                
                metrics['edge_level']['supervised'] = {
                    'auroc': float(roc_auc_score(labels, scores)),
                    'auprc': float(average_precision_score(labels, scores)),
                }
                
                # Find optimal threshold for F1 score
                from sklearn.metrics import f1_score
                best_f1 = 0
                best_threshold = np.median(scores)
                best_predictions = predictions
                
                # Try different percentile thresholds
                for percentile in [50, 75, 90, 95, 99]:
                    threshold = np.percentile(scores, percentile)
                    temp_predictions = (np.array(scores) > threshold).astype(int)
                    temp_f1 = f1_score(labels, temp_predictions, zero_division=0)
                    if temp_f1 > best_f1:
                        best_f1 = temp_f1
                        best_threshold = threshold
                        best_predictions = temp_predictions
                
                # Use best predictions
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, best_predictions, average='binary', zero_division=0
                )
                metrics['edge_level']['supervised']['precision'] = float(precision)
                metrics['edge_level']['supervised']['recall'] = float(recall)
                metrics['edge_level']['supervised']['f1_score'] = float(f1)
                metrics['edge_level']['supervised']['threshold'] = float(best_threshold)
                
                cm = confusion_matrix(labels, predictions)
                metrics['edge_level']['supervised']['confusion_matrix'] = cm.tolist()
                
                logger.info(f"Edge-level supervised metrics: AUROC={metrics['edge_level']['supervised']['auroc']:.4f}, "
                          f"F1={metrics['edge_level']['supervised']['f1_score']:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate edge-level supervised metrics: {e}")
                metrics['edge_level']['supervised'] = None
        else:
            metrics['edge_level']['supervised'] = None
        
        # ========== ENTITY-LEVEL METRICS (Paper-style aggregation) ==========
        
        # Entity-level aggregation: group edges by source entity and aggregate scores
        # This matches the methodology in MAGIC/Kairos papers (Table 3)
        logger.info("Computing entity-level aggregation (matches paper methodology)")
        
        # Try to get edge information from batch construction or featurization
        entity_level_available = False
        
        try:
            # Get batch construction data to extract edge->entity mapping
            if 'batch_construction' in dependencies:
                batch_data = dependencies['batch_construction']
                data_list = batch_data.get('data_list', [])
                
                # Extract edge index (source, destination) for entity mapping
                entity_scores = {}  # entity_id -> list of scores
                entity_labels = {}  # entity_id -> label (if available)
                edge_idx = 0
                
                for data in data_list:
                    if hasattr(data, 'edge_index'):
                        edge_index = data.edge_index
                        num_edges_in_batch = edge_index.shape[1]
                        
                        # Get scores and labels for this batch
                        batch_scores = scores_array[edge_idx:edge_idx + num_edges_in_batch]
                        batch_labels = labels[edge_idx:edge_idx + num_edges_in_batch] if labels is not None else None
                        
                        # Aggregate by source entity
                        for i in range(num_edges_in_batch):
                            src_entity = int(edge_index[0, i])  # Source node ID
                            
                            if src_entity not in entity_scores:
                                entity_scores[src_entity] = []
                            entity_scores[src_entity].append(float(batch_scores[i]))
                            
                            # Store entity label (majority vote if multiple edges)
                            if batch_labels is not None:
                                if src_entity not in entity_labels:
                                    entity_labels[src_entity] = []
                                entity_labels[src_entity].append(int(batch_labels[i]))
                        
                        edge_idx += num_edges_in_batch
                
                if entity_scores:
                    entity_level_available = True
                    
                    # Aggregate scores: use MAX score per entity (most anomalous edge)
                    entity_max_scores = {eid: max(scores_list) for eid, scores_list in entity_scores.items()}
                    entity_mean_scores = {eid: np.mean(scores_list) for eid, scores_list in entity_scores.items()}
                    
                    # Aggregate labels: use majority vote (or max for malicious)
                    entity_final_labels = {}
                    if entity_labels:
                        for eid, label_list in entity_labels.items():
                            # If any edge is malicious (1), entity is malicious
                            entity_final_labels[eid] = max(label_list)
                    
                    # Convert to arrays
                    entity_ids = sorted(entity_max_scores.keys())
                    entity_scores_array = np.array([entity_max_scores[eid] for eid in entity_ids])
                    entity_labels_array = np.array([entity_final_labels[eid] for eid in entity_ids]) if entity_final_labels else None
                    
                    # Calculate entity-level metrics
                    metrics['entity_level'] = {}
                    metrics['entity_level']['num_entities'] = len(entity_ids)
                    metrics['entity_level']['aggregation_method'] = 'max'  # MAX score per entity
                    
                    metrics['entity_level']['anomaly_score_stats'] = {
                        'mean': float(np.mean(entity_scores_array)),
                        'std': float(np.std(entity_scores_array)),
                        'min': float(np.min(entity_scores_array)),
                        'max': float(np.max(entity_scores_array)),
                        'median': float(np.median(entity_scores_array)),
                    }
                    
                    # Entity-level supervised metrics (if labels available)
                    if entity_labels_array is not None and len(np.unique(entity_labels_array)) > 1:
                        try:
                            from sklearn.metrics import (
                                roc_auc_score, average_precision_score,
                                precision_recall_fscore_support
                            )
                            
                            metrics['entity_level']['supervised'] = {
                                'auroc': float(roc_auc_score(entity_labels_array, entity_scores_array)),
                                'auprc': float(average_precision_score(entity_labels_array, entity_scores_array)),
                            }
                            
                            # Threshold at median for predictions
                            threshold = np.median(entity_scores_array)
                            entity_predictions = (entity_scores_array > threshold).astype(int)
                            
                            precision, recall, f1, support = precision_recall_fscore_support(
                                entity_labels_array, entity_predictions, average='binary', zero_division=0
                            )
                            metrics['entity_level']['supervised']['precision'] = float(precision)
                            metrics['entity_level']['supervised']['recall'] = float(recall)
                            metrics['entity_level']['supervised']['f1_score'] = float(f1)
                            
                            logger.info(f"✓ Entity-level supervised metrics (n={len(entity_ids)}):")
                            logger.info(f"  AUROC: {metrics['entity_level']['supervised']['auroc']:.4f}")
                            logger.info(f"  AUPRC: {metrics['entity_level']['supervised']['auprc']:.4f}")
                            logger.info(f"  F1: {metrics['entity_level']['supervised']['f1_score']:.4f}")
                            logger.info(f"  (Aggregation: MAX score per entity)")
                        except Exception as e:
                            logger.warning(f"Could not calculate entity-level supervised metrics: {e}")
                            metrics['entity_level']['supervised'] = None
                    else:
                        metrics['entity_level']['supervised'] = None
                else:
                    logger.warning("No entities extracted from batch data")
        
        except Exception as e:
            logger.warning(f"Could not perform entity-level aggregation: {e}")
            import traceback
            traceback.print_exc()
        
        if not entity_level_available:
            metrics['entity_level'] = {'available': False, 'reason': 'Could not extract entity information'}
            logger.warning("Entity-level metrics not available - using edge-level only")
        
        # ========== SUMMARY ==========
        
        metrics['num_samples'] = {'edges': len(scores_array)}
        if entity_level_available:
            metrics['num_samples']['entities'] = metrics['entity_level']['num_entities']
        
        metrics['detection_approach'] = 'unsupervised'
        
        # Log edge-level summary
        logger.info(f"Edge-Level Anomaly Detection Metrics:")
        logger.info(f"  Score range: [{metrics['edge_level']['anomaly_score_stats']['min']:.6f}, {metrics['edge_level']['anomaly_score_stats']['max']:.6f}]")
        logger.info(f"  Mean: {metrics['edge_level']['anomaly_score_stats']['mean']:.6f}, Std: {metrics['edge_level']['anomaly_score_stats']['std']:.6f}")
        logger.info(f"  Separation ratio: {metrics['edge_level']['score_separation_ratio']:.4f}")
        logger.info(f"  Critical anomalies (99.9%): {metrics['edge_level']['anomaly_counts']['critical_99.9']}")
        
        return metrics
    
    @staticmethod
    def attack_tracing(
        config: Dict[str, Any],
        task_config: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 9: Post-processing attack tracing (optional).
        
        Traces back from detected anomalies to identify attack paths
        and entry points.
        
        Args:
            config: Global configuration
            task_config: Tracing config
            dependencies: Inference and metrics results
            
        Returns:
            Attack trace information
        """
        enabled = task_config.get('enabled', False)
        
        if not enabled:
            logger.info("Attack tracing disabled, skipping")
            return {'enabled': False}
        
        logger.info("Performing attack tracing (placeholder)")
        
        # This is a placeholder for attack tracing functionality
        # Full implementation would trace backward from detected anomalies
        
        return {
            'enabled': True,
            'message': 'Attack tracing not yet implemented'
        }
