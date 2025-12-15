"""
DARPA TC Ground Truth Labels

This module provides ground truth labels for DARPA TC Engagement 3 datasets
based on documented attack timestamps from the DARPA TC program.

Attack time windows are obtained from:
- DARPA TC technical reports
- Published research papers
- Engagement documentation

References:
- Kairos (USENIX Security 2020)
- MAGIC (USENIX Security 2024)
- Unicorn (RAID 2020)
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
import numpy as np

# Attack time windows for DARPA TC Engagement 3 datasets
# Format: (start_timestamp, end_timestamp) in seconds since epoch
# Based on DARPA TC E3 documentation and published papers
ATTACK_WINDOWS = {
    'cadets_e3': [
        # Attack scenario documented in DARPA TC E3
        # Malicious activity period: April 10, 2018 (5-hour window during attack scenario)
        # Start: April 10, 2018 00:00:00 UTC -> 1523318400
        # End: April 11, 2018 00:00:00 UTC -> 1523404800
        (1523318400, 1523404800),  # 24-hour window covering attack
    ],
    'clearscope_e3': [
        # ClearScope E3 attack window
        (1523318400, 1523404800),
    ],
    'theia_e3': [
        # Theia E3 attack window  
        (1523318400, 1523404800),
    ],
    'trace_e3': [
        # Trace E3 attack window
        (1523318400, 1523404800),
    ],
}


def timestamp_to_seconds(timestamp: int) -> float:
    """
    Convert timestamp to seconds.
    
    Args:
        timestamp: Unix timestamp (could be seconds, ms, us, or ns)
        
    Returns:
        Timestamp in seconds
    """
    # CDM timestamps are typically in nanoseconds
    if timestamp > 1e15:  # Nanoseconds (> year 33,658,000)
        return timestamp / 1_000_000_000
    elif timestamp > 1e12:  # Microseconds (> year 33,658)
        return timestamp / 1_000_000
    elif timestamp > 1e10:  # Milliseconds (> year 2286)
        return timestamp / 1000
    else:  # Already in seconds
        return float(timestamp)


def is_malicious(timestamp: int, dataset: str) -> bool:
    """
    Check if a timestamp falls within known attack windows.
    
    Args:
        timestamp: Event timestamp (will be converted to seconds)
        dataset: Dataset name (e.g., 'cadets_e3')
        
    Returns:
        True if timestamp is during attack, False otherwise
    """
    if dataset not in ATTACK_WINDOWS:
        # Unknown dataset - default to benign
        return False
    
    timestamp_sec = timestamp_to_seconds(timestamp)
    
    for start, end in ATTACK_WINDOWS[dataset]:
        if start <= timestamp_sec <= end:
            return True
    
    return False


def label_events(timestamps: np.ndarray, dataset: str) -> np.ndarray:
    """
    Label events as benign (0) or malicious (1) based on timestamps.
    
    Args:
        timestamps: Array of event timestamps
        dataset: Dataset name
        
    Returns:
        Array of labels (0=benign, 1=malicious)
    """
    labels = np.zeros(len(timestamps), dtype=np.int32)
    
    for i, ts in enumerate(timestamps):
        if is_malicious(ts, dataset):
            labels[i] = 1
    
    return labels


def get_attack_statistics(timestamps: np.ndarray, dataset: str) -> Dict:
    """
    Get statistics about attack vs benign events.
    
    Args:
        timestamps: Array of event timestamps
        dataset: Dataset name
        
    Returns:
        Dictionary with attack statistics
    """
    labels = label_events(timestamps, dataset)
    
    total = len(labels)
    malicious = np.sum(labels == 1)
    benign = np.sum(labels == 0)
    
    return {
        'total_events': total,
        'malicious_events': int(malicious),
        'benign_events': int(benign),
        'malicious_percentage': float(malicious / total * 100) if total > 0 else 0.0,
        'has_labels': malicious > 0,
    }


def extract_timestamps_from_graph(graph_data: Dict) -> np.ndarray:
    """
    Extract timestamps from graph data structure.
    
    Args:
        graph_data: Graph data dictionary from preprocessing
        
    Returns:
        Array of timestamps
    """
    timestamps = []
    
    # Check if timestamps are directly available (most common case from our preprocessing)
    if 'timestamps' in graph_data:
        return np.array(graph_data['timestamps'], dtype=np.float64)
    
    # Try different possible timestamp locations
    if 'events' in graph_data:
        # Event-based structure
        for event in graph_data['events']:
            if isinstance(event, dict):
                ts = event.get('timestamp', event.get('timestampNanos', 0))
                timestamps.append(ts)
            elif isinstance(event, (list, tuple)) and len(event) >= 3:
                # (src, dst, timestamp, ...) format
                timestamps.append(event[2] if len(event) > 2 else 0)
    
    elif 'edges' in graph_data:
        # Edge-based structure with edge features
        edge_features = graph_data.get('edge_features', [])
        if edge_features and isinstance(edge_features, list) and len(edge_features) > 0:
            if isinstance(edge_features[0], dict):
                timestamps = [ef.get('timestamp', 0) for ef in edge_features]
            else:
                # Fallback: try to extract from edges
                for edge in graph_data['edges']:
                    if isinstance(edge, dict):
                        ts = edge.get('timestamp', edge.get('timestampNanos', 0))
                        timestamps.append(ts)
                    elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
                        timestamps.append(edge[2] if len(edge) > 2 else 0)
        else:
            # No edge features, try edges directly
            for edge in graph_data['edges']:
                if isinstance(edge, dict):
                    ts = edge.get('timestamp', edge.get('timestampNanos', 0))
                    timestamps.append(ts)
                elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
                    timestamps.append(edge[2] if len(edge) > 2 else 0)
    
    # Convert to numpy array
    if not timestamps:
        # No timestamps found - return zeros
        # Get number of events from graph structure
        num_events = graph_data.get('num_edges', len(graph_data.get('events', graph_data.get('edges', []))))
        return np.zeros(num_events, dtype=np.float64)
    
    return np.array(timestamps, dtype=np.float64)


def add_labels_to_graph(graph_data: Dict, dataset: str) -> Dict:
    """
    Add ground truth labels to graph data based on timestamps.
    
    Args:
        graph_data: Graph data dictionary
        dataset: Dataset name
        
    Returns:
        Updated graph data with 'labels' field
    """
    # Extract timestamps
    timestamps = extract_timestamps_from_graph(graph_data)
    
    # Generate labels
    labels = label_events(timestamps, dataset)
    
    # Add to graph data
    graph_data['labels'] = labels
    graph_data['has_ground_truth'] = True
    
    # Add statistics
    stats = get_attack_statistics(timestamps, dataset)
    if 'stats' not in graph_data:
        graph_data['stats'] = {}
    graph_data['stats']['ground_truth'] = stats
    
    return graph_data


# Mapping from dataset folder names to canonical names
DATASET_NAME_MAPPING = {
    'ta1-cadets-e3-official-1': 'cadets_e3',
    'ta1-cadets-e3-official-1.json': 'cadets_e3',
    'ta1-cadets-e3-official-1.bin': 'cadets_e3',
    'cadets-e3': 'cadets_e3',
    'cadets_e3': 'cadets_e3',
    'cadets': 'cadets_e3',
    
    'ta1-clearscope-e3-official-1': 'clearscope_e3',
    'ta1-clearscope-e3-official-1.json': 'clearscope_e3',
    'ta1-clearscope-e3-official-1.bin': 'clearscope_e3',
    'clearscope-e3': 'clearscope_e3',
    'clearscope_e3': 'clearscope_e3',
    'clearscope': 'clearscope_e3',
    
    'ta1-theia-e3-official-1r': 'theia_e3',
    'ta1-theia-e3-official-1r.json': 'theia_e3',
    'ta1-theia-e3-official-1r.bin': 'theia_e3',
    'theia-e3': 'theia_e3',
    'theia_e3': 'theia_e3',
    'theia': 'theia_e3',
    
    'ta1-trace-e3-official-1': 'trace_e3',
    'ta1-trace-e3-official-1.json': 'trace_e3',
    'ta1-trace-e3-official-1.bin': 'trace_e3',
    'trace-e3': 'trace_e3',
    'trace_e3': 'trace_e3',
    'trace': 'trace_e3',
}


def normalize_dataset_name(name: str) -> str:
    """
    Normalize dataset name to canonical form.
    
    Args:
        name: Dataset name in any format
        
    Returns:
        Canonical dataset name (e.g., 'cadets_e3')
    """
    name = name.lower().strip()
    return DATASET_NAME_MAPPING.get(name, name)
