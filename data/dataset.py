"""
Base dataset class for PIDS framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class BasePIDSDataset(ABC, Dataset):
    """
    Abstract base class for PIDS datasets.
    """
    
    def __init__(self, data_path: Path, config: Dict[str, Any], split: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset
            config: Dataset configuration
            split: 'train', 'val', or 'test'
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.data = []
        self.labels = []
        
        self.load_data()
    
    @abstractmethod
    def load_data(self):
        """Load and preprocess data."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int):
        """Get item by index."""
        pass
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'num_samples': len(self),
            'num_classes': len(set(self.labels)) if self.labels else 0,
            'split': self.split,
            'data_path': str(self.data_path)
        }


class GraphDataset(BasePIDSDataset):
    """Dataset for graph-based PIDS models."""
    
    def __init__(self, data_path: Path, config: Dict[str, Any], split: str = 'train'):
        self.graphs = []
        super().__init__(data_path, config, split)
    
    def load_data(self):
        """Load graph data."""
        logger.info(f"Loading graph dataset from {self.data_path} ({self.split} split)")
        # Implementation depends on specific format
        # This is a placeholder
        pass
    
    def __getitem__(self, idx: int):
        """Get graph by index."""
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self) -> int:
        return len(self.graphs)


class StreamDataset(BasePIDSDataset):
    """Dataset for streaming/temporal data."""
    
    def load_data(self):
        """Load streaming data."""
        logger.info(f"Loading stream dataset from {self.data_path} ({self.split} split)")
        # Implementation placeholder
        pass
    
    def __getitem__(self, idx: int):
        """Get item by index."""
        return self.data[idx], self.labels[idx]


class CustomSOCDataset(BasePIDSDataset):
    """
    Dataset for custom SOC logs.
    Handles JSON format from Elastic/ELK stack.
    """
    
    def __init__(self, data_path: Path, config: Dict[str, Any], split: str = 'train'):
        self.events = []
        self.event_types = ['process', 'file', 'network']
        super().__init__(data_path, config, split)
    
    def load_data(self):
        """Load custom SOC data from JSON files."""
        import json
        from tqdm import tqdm
        
        logger.info(f"Loading custom SOC dataset from {self.data_path}")
        
        # Load different event types
        for event_type in self.event_types:
            file_path = self.data_path / f"endpoint_{event_type}.json"
            if file_path.exists():
                logger.info(f"Loading {event_type} events from {file_path}")
                try:
                    with open(file_path, 'r') as f:
                        # Read large JSON file in chunks
                        events = json.load(f)
                        logger.info(f"Loaded {len(events)} {event_type} events")
                        self.events.extend(events)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total events loaded: {len(self.events)}")
        
        # Process events into graphs or sequences
        self.data = self.process_events()
        
        # For now, all samples are labeled as benign (0)
        # In practice, you would need ground truth labels
        self.labels = [0] * len(self.data)
    
    def process_events(self) -> List[Any]:
        """
        Process raw events into model input format.
        This should construct provenance graphs from events.
        """
        logger.info("Processing events into model input format...")
        
        # Placeholder: group events by time window
        from collections import defaultdict
        
        # Group by endpoint and time window
        time_window = self.config.get('time_window', 3600)  # 1 hour default
        grouped_events = defaultdict(list)
        
        for event in self.events:
            try:
                timestamp = event.get('@timestamp', '')
                endpoint_id = event.get('agent', {}).get('id', 'unknown')
                
                # Simple grouping - in practice, use proper time bucketing
                key = f"{endpoint_id}_{timestamp[:13]}"  # Hour-level grouping
                grouped_events[key].append(event)
            except Exception as e:
                logger.warning(f"Error processing event: {e}")
                continue
        
        # Convert grouped events to graph or sequence format
        processed_data = []
        for key, events in grouped_events.items():
            # Placeholder: each group becomes one sample
            # In practice, construct provenance graph from events
            processed_data.append({
                'key': key,
                'events': events,
                'num_events': len(events)
            })
        
        logger.info(f"Processed into {len(processed_data)} samples")
        return processed_data
    
    def __getitem__(self, idx: int):
        """Get processed sample by index."""
        return self.data[idx], self.labels[idx]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'total_events': len(self.events),
            'event_types': self.event_types,
            'num_samples': len(self.data)
        })
        return metadata


def get_dataset(dataset_name: str, data_path: Path, config: Dict[str, Any], 
                split: str = 'train') -> BasePIDSDataset:
    """
    Factory function to get appropriate dataset.
    
    Args:
        dataset_name: Name of dataset
        data_path: Path to dataset
        config: Dataset configuration
        split: Data split
        
    Returns:
        Dataset instance
    """
    dataset_map = {
        'custom': CustomSOCDataset,
        'graph': GraphDataset,
        'stream': StreamDataset,
    }
    
    if dataset_name not in dataset_map:
        logger.warning(f"Unknown dataset type: {dataset_name}, using CustomSOCDataset")
        dataset_class = CustomSOCDataset
    else:
        dataset_class = dataset_map[dataset_name]
    
    return dataset_class(data_path, config, split)
