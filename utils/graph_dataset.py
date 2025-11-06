"""
Graph dataset loader for training.

This module provides a simple dataset class that loads preprocessed graph data
created by preprocess_data.py for training PIDS models.
"""
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class PreprocessedGraphDataset(Dataset):
    """
    Dataset class for loading preprocessed provenance graphs.
    
    This class loads graphs that have been preprocessed by preprocess_data.py
    and converts them to PyTorch Geometric format for training.
    """
    
    def __init__(
        self,
        graph_path: Path,
        mode: str = 'unsupervised',
        labels_path: Optional[Path] = None
    ):
        """
        Initialize dataset from preprocessed graph file.
        
        Args:
            graph_path: Path to preprocessed graph pickle file
            mode: Training mode ('supervised', 'unsupervised', 'semi_supervised')
            labels_path: Optional path to labels file for supervised training
        """
        self.graph_path = Path(graph_path)
        self.mode = mode
        self.labels_path = labels_path
        
        self.graphs = []
        self.labels = []
        self.metadata = {}
        
        # Load graph data
        self.load_graph_data()
        
        # Load labels if supervised
        if mode in ['supervised', 'semi_supervised'] and labels_path:
            self.load_labels()
    
    def load_graph_data(self):
        """Load preprocessed graph from pickle file."""
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
        
        logger.info(f"Loading preprocessed graph from {self.graph_path}")
        
        try:
            with open(self.graph_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Load metadata from JSON if available
            stats_path = self.graph_path.with_suffix('.json')
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Convert graph data to PyTorch Geometric format
            self.graphs = self._convert_to_pyg_graphs(graph_data)
            
            logger.info(f"✓ Loaded {len(self.graphs)} graphs")
            logger.info(f"  Nodes: {self.metadata.get('num_nodes', 'N/A'):,}")
            logger.info(f"  Edges: {self.metadata.get('num_edges', 'N/A'):,}")
            
        except Exception as e:
            logger.error(f"Error loading graph data: {e}")
            raise
    
    def _convert_to_pyg_graphs(self, graph_data) -> List[Data]:
        """
        Convert preprocessed graph data to PyTorch Geometric Data objects.
        
        Handles both formats:
        - Single graph (dict): Old format, single large graph
        - Multiple graphs (list): New format with time-windowed graphs
        """
        graphs = []
        
        # Check if we have a list of graphs (time windows) or a single graph
        if isinstance(graph_data, list):
            # New format: list of time-windowed graphs
            logger.info(f"Processing {len(graph_data)} time-windowed graphs")
            for i, single_graph in enumerate(graph_data):
                pyg_graph = self._convert_single_graph_to_pyg(single_graph)
                if pyg_graph is not None:
                    graphs.append(pyg_graph)
        else:
            # Old format: single large graph (dict)
            logger.info("Processing single large graph")
            pyg_graph = self._convert_single_graph_to_pyg(graph_data)
            if pyg_graph is not None:
                graphs.append(pyg_graph)
        
        return graphs
    
    def _convert_single_graph_to_pyg(self, graph_data: Dict) -> Optional[Data]:
        """Convert a single graph dict to PyTorch Geometric Data object."""
        # Extract graph components
        edges = graph_data.get('edges', [])
        num_nodes = graph_data.get('num_nodes', 0)
        node_features = graph_data.get('node_features', {})
        edge_features = graph_data.get('edge_features', [])
        timestamps = graph_data.get('timestamps', [])
        node_type_map = graph_data.get('node_type_map', {})
        edge_type_map = graph_data.get('edge_type_map', {})
        
        if not edges or num_nodes == 0:
            logger.warning("Graph data is empty")
            return None
        
        # Build node feature matrix
        x = self._build_node_features(num_nodes, node_type_map, node_features)
        
        # Build edge index
        edge_index = torch.tensor(
            [[e[0] for e in edges], [e[1] for e in edges]], 
            dtype=torch.long
        )
        
        # Build edge attributes (if available)
        edge_attr = None
        if edge_features:
            edge_attr = self._build_edge_features(edge_features, edge_type_map)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        # Store additional metadata
        data.timestamps = timestamps if timestamps else None
        data.node_type_map = node_type_map
        data.edge_type_map = edge_type_map
        
        return data
    
    def _build_node_features(
        self, 
        num_nodes: int, 
        node_type_map: Dict, 
        node_features: Dict
    ) -> torch.Tensor:
        """Build node feature matrix."""
        # Create basic feature matrix with node type encoding
        feature_dim = 128  # Default feature dimension
        x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        
        # Entity type encoding
        entity_type_map = {
            'process': 1.0,
            'file': 2.0,
            'network': 3.0,
            'socket': 4.0,
            'memory': 5.0,
            'registry': 6.0,
            'principal': 7.0,
        }
        
        for node_id, node_type in node_type_map.items():
            if isinstance(node_id, str):
                node_id = int(node_id)
            x[node_id, 0] = entity_type_map.get(node_type, 0.0)
            
            # Add additional features if available
            if node_id in node_features and node_features[node_id]:
                features = node_features[node_id]
                if isinstance(features, list):
                    # Copy available features
                    feat_len = min(len(features), feature_dim - 1)
                    for i, feat_val in enumerate(features[:feat_len]):
                        if isinstance(feat_val, (int, float)):
                            x[node_id, i + 1] = float(feat_val)
        
        return x
    
    def _build_edge_features(
        self,
        edge_features: List,
        edge_type_map: Dict
    ) -> torch.Tensor:
        """Build edge feature matrix."""
        feature_dim = 64  # Default edge feature dimension
        num_edges = len(edge_features)
        
        edge_attr = torch.zeros((num_edges, feature_dim), dtype=torch.float)
        
        # Event type encoding
        event_type_map = {
            'exec': 1.0,
            'fork': 2.0,
            'read': 3.0,
            'write': 4.0,
            'open': 5.0,
            'close': 6.0,
            'connect': 7.0,
            'accept': 8.0,
            'send': 9.0,
            'recv': 10.0,
        }
        
        for i, edge_feat in enumerate(edge_features):
            if isinstance(edge_feat, dict):
                # Extract event type
                event_type = edge_feat.get('event_type', 'unknown')
                edge_attr[i, 0] = event_type_map.get(event_type, 0.0)
                
                # Extract timestamp (normalized to time of day)
                timestamp = edge_feat.get('timestamp', 0)
                if timestamp > 0:
                    edge_attr[i, 1] = (timestamp % 86400) / 86400.0
        
        return edge_attr
    
    def load_labels(self):
        """Load ground truth labels for supervised training."""
        if not self.labels_path or not self.labels_path.exists():
            logger.warning(f"Labels file not found: {self.labels_path}")
            logger.warning("Using default labels (all benign)")
            self.labels = [0] * len(self.graphs)
            return
        
        logger.info(f"Loading labels from {self.labels_path}")
        
        try:
            with open(self.labels_path, 'r') as f:
                labels_data = json.load(f)
            
            # Extract labels (format depends on ground truth structure)
            if isinstance(labels_data, list):
                self.labels = labels_data
            elif isinstance(labels_data, dict):
                # Map graph IDs to labels
                self.labels = [labels_data.get(str(i), 0) for i in range(len(self.graphs))]
            
            logger.info(f"✓ Loaded {len(self.labels)} labels")
            
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            self.labels = [0] * len(self.graphs)
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get graph by index.
        
        Args:
            idx: Index of graph
            
        Returns:
            PyTorch Geometric Data object
        """
        graph = self.graphs[idx]
        
        # Add label for supervised training
        if self.mode in ['supervised', 'semi_supervised']:
            label = self.labels[idx] if idx < len(self.labels) else 0
            graph.y = torch.tensor([label], dtype=torch.long)
        
        return graph
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'num_samples': len(self),
            'mode': self.mode,
            'graph_path': str(self.graph_path),
            **self.metadata
        }
