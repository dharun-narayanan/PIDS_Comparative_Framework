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
        
        # Look for preprocessed pickle file
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
        for streaming graph processing.
        
        Args:
            config: Global configuration
            task_config: Task config (window_size, overlap, etc.)
            dependencies: Must contain 'load_preprocessed_data' result
            
        Returns:
            List of time-windowed graphs
        """
        graph_data = dependencies['load_preprocessed_data']['graph_data']
        
        window_size = task_config.get('window_size', 3600)  # seconds
        overlap = task_config.get('overlap', 0.0)  # fraction
        
        logger.info(f"Constructing time windows (size={window_size}s, overlap={overlap})")
        
        edges = graph_data.get('edges', graph_data.get('events', []))
        
        # Sort by timestamp
        if edges and 'timestamp' in edges[0]:
            edges = sorted(edges, key=lambda x: x['timestamp'])
        
        # Create time windows
        time_windows = []
        if edges:
            min_time = edges[0].get('timestamp', 0)
            max_time = edges[-1].get('timestamp', 0)
            
            current_start = min_time
            stride = window_size * (1 - overlap)
            
            while current_start < max_time:
                current_end = current_start + window_size
                
                # Get edges in this window
                window_edges = [
                    e for e in edges
                    if current_start <= e.get('timestamp', 0) < current_end
                ]
                
                if window_edges:
                    time_windows.append({
                        'start_time': current_start,
                        'end_time': current_end,
                        'edges': window_edges,
                        'num_edges': len(window_edges)
                    })
                
                current_start += stride
        
        logger.info(f"Created {len(time_windows)} time windows")
        
        return {
            'time_windows': time_windows,
            'window_size': window_size,
            'overlap': overlap,
            'original_graph': graph_data
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
            graph_data = dependencies['load_preprocessed_data']['graph_data']
            time_windows = [{'edges': graph_data.get('edges', []), 'original': True}]
        
        transform_type = task_config.get('type', 'none')
        
        logger.info(f"Applying graph transformation: {transform_type}")
        
        transformed_windows = []
        for window in time_windows:
            edges = window['edges']
            
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
        
        # Create reverse mapping for edge types (id -> name)
        edge_id_to_type = {v: k for k, v in edge_type_map.items()} if edge_type_map else {}
        
        method = task_config.get('method', 'one_hot')
        node_feat_dim = task_config.get('node_feat_dim', 128)
        edge_feat_dim = task_config.get('edge_feat_dim', 64)
        
        logger.info(f"Extracting features using method: {method}")
        
        # Collect all unique nodes and edges
        all_nodes = set()
        all_edge_types = set()
        node_types = {}
        
        for window in windows:
            for edge in window.get('edges', []):
                # Handle both tuple and dict formats
                if isinstance(edge, tuple):
                    # Tuple format: (src_id, dst_id, edge_type_id)
                    src, dst, edge_type_id = edge
                    all_nodes.add(src)
                    all_nodes.add(dst)
                    if src in node_type_map:
                        node_types[src] = node_type_map[src]
                    if dst in node_type_map:
                        node_types[dst] = node_type_map[dst]
                    if edge_type_id in edge_id_to_type:
                        all_edge_types.add(edge_id_to_type[edge_type_id])
                else:
                    # Dictionary format
                    src = edge.get('src')
                    dst = edge.get('dst')
                    if src:
                        all_nodes.add(src)
                        if 'src_type' in edge:
                            node_types[src] = edge['src_type']
                    if dst:
                        all_nodes.add(dst)
                        if 'dst_type' in edge:
                            node_types[dst] = edge['dst_type']
                    if 'type' in edge:
                        all_edge_types.add(edge['type'])
        
        num_nodes = len(all_nodes)
        num_edge_types = len(all_edge_types)
        
        logger.info(f"Found {num_nodes} unique nodes, {num_edge_types} edge types")
        
        # Create node ID mapping
        node_to_id = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        edge_type_to_id = {et: idx for idx, et in enumerate(sorted(all_edge_types))}
        
        # Initialize feature matrices
        if method == 'one_hot':
            # One-hot encoding based on node types
            unique_node_types = set(node_types.values())
            node_type_to_id = {nt: idx for idx, nt in enumerate(sorted(unique_node_types))}
            
            node_features = np.zeros((num_nodes, len(unique_node_types)))
            for node, node_id in node_to_id.items():
                if node in node_types:
                    type_id = node_type_to_id[node_types[node]]
                    node_features[node_id, type_id] = 1.0
            
            edge_features = np.eye(num_edge_types)  # One-hot for edge types
        
        elif method == 'random':
            # Random features (baseline)
            node_features = np.random.randn(num_nodes, node_feat_dim)
            edge_features = np.random.randn(num_edge_types, edge_feat_dim)
        
        elif method == 'pretrained':
            # Load pretrained embeddings if available
            embed_path = task_config.get('embedding_path')
            if embed_path and Path(embed_path).exists():
                with open(embed_path, 'rb') as f:
                    embeddings = pickle.load(f)
                node_features = embeddings.get('node_features', np.random.randn(num_nodes, node_feat_dim))
                edge_features = embeddings.get('edge_features', np.eye(num_edge_types))
            else:
                logger.warning("Pretrained embeddings not found, using random")
                node_features = np.random.randn(num_nodes, node_feat_dim)
                edge_features = np.eye(num_edge_types)
        
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
        
        Args:
            config: Global configuration
            task_config: Batch construction config
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
        
        logger.info(f"Constructing batches (batch_size={batch_size})")
        
        # Get metadata for tuple-format edges from feature data (passed through the pipeline)
        graph_data = feat_data.get('graph_data', {})
        edge_type_map = graph_data.get('edge_type_map', {})
        edge_id_to_type = {v: k for k, v in edge_type_map.items()} if edge_type_map else {}
        
        # Create Data objects for each window
        data_list = []
        for i, window in enumerate(windows):
            edges = window.get('edges', [])
            
            if not edges:
                continue
            
            # Build edge index
            edge_index = []
            edge_attrs = []
            edge_labels = []
            
            for edge in edges:
                # Handle both tuple and dict formats
                if isinstance(edge, tuple):
                    # Tuple format: (src_id, dst_id, edge_type_id)
                    src, dst, edge_type_id = edge
                    edge_type = edge_id_to_type.get(edge_type_id, 'unknown')
                    label = 0  # Default label for tuple format
                else:
                    # Dictionary format
                    src = edge.get('src')
                    dst = edge.get('dst')
                    edge_type = edge.get('type', 'unknown')
                    label = edge.get('label', 0)
                
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
                data = Data(
                    x=node_features,
                    edge_index=torch.LongTensor(edge_index).t().contiguous(),
                    edge_attr=torch.stack(edge_attrs) if edge_attrs else None,
                    y=torch.LongTensor(edge_labels),
                    num_nodes=len(node_to_id)
                )
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
            checkpoint_path = checkpoint_template.format(model_name=model_name, dataset=dataset_name)
            
            # Check if checkpoint exists
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                # Try alternatives
                checkpoint_dir = Path('checkpoints') / model_name
                alternatives = [
                    checkpoint_dir / f'checkpoint-{dataset_name}.pt',
                    checkpoint_dir / f'checkpoint-{model_name}.pt',
                    checkpoint_dir / 'best.pt',
                    checkpoint_dir / 'checkpoint.pt',
                ]
                for alt in alternatives:
                    if alt.exists():
                        checkpoint_path = str(alt)
                        logger.info(f"Using alternative checkpoint: {checkpoint_path}")
                        break
                else:
                    checkpoint_path = None
        
        # Build and load model
        logger.info(f"Building model: {model_name}")
        model = model_builder.build_and_load(
            model_name,
            dataset_name=dataset_name,
            checkpoint_path=checkpoint_path,
            device=device
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
        
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch, inference=True)
                
                # Handle different output formats
                if isinstance(output, dict):
                    # Multi-decoder output, use primary decoder
                    primary_key = list(output.keys())[0]
                    output = output[primary_key]
                
                if isinstance(output, torch.Tensor):
                    if output.dim() == 2 and output.shape[1] > 1:
                        # Multi-class classification (already softmax from decoder)
                        scores = output
                        predictions = torch.argmax(scores, dim=1)
                    else:
                        # Binary classification (already sigmoid from decoder)
                        scores = output.squeeze()
                        predictions = (scores > 0.5).long()
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
        Task 8: Calculate evaluation metrics.
        
        Computes standard detection metrics:
        - AUROC, AUPRC
        - Precision, Recall, F1
        - Confusion Matrix
        - Detection rate at different FPR thresholds
        
        Args:
            config: Global configuration
            task_config: Metrics config
            dependencies: Model inference results
            
        Returns:
            Dictionary of computed metrics
        """
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_recall_fscore_support, confusion_matrix
        )
        
        inference_results = dependencies['model_inference']
        
        predictions = inference_results['predictions']
        scores = inference_results['scores']
        labels = inference_results['labels']
        
        if labels is None:
            logger.warning("No ground truth labels available, skipping metrics")
            return {'message': 'No labels available'}
        
        logger.info("Calculating evaluation metrics")
        
        metrics = {}
        
        # ROC-AUC and PR-AUC
        try:
            if len(np.unique(labels)) > 1:  # Need both classes
                metrics['auroc'] = roc_auc_score(labels, scores)
                metrics['auprc'] = average_precision_score(labels, scores)
            else:
                logger.warning("Only one class present in labels, skipping AUC metrics")
                metrics['auroc'] = 0.0
                metrics['auprc'] = 0.0
        except Exception as e:
            logger.warning(f"Error calculating AUC metrics: {e}")
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional stats
        metrics['num_samples'] = len(labels)
        metrics['num_positive'] = int(np.sum(labels))
        metrics['num_negative'] = int(len(labels) - np.sum(labels))
        metrics['prediction_positive'] = int(np.sum(predictions))
        
        logger.info(f"Metrics: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_score']:.4f}")
        
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
