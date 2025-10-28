#!/usr/bin/env python3
"""
Preprocess custom SOC data for PIDS models.

This script converts custom Security Operations Center logs into graph format
suitable for training PIDS models.
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.common import setup_logging, load_config
import logging

logger = setup_logging()
logger = logging.getLogger("preprocess_data")


class SOCDataPreprocessor:
    """Preprocessor for custom SOC data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get('data', {})
        self.format_config = config.get('format', {})
        self.graph_config = config.get('graph', {})
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Entity mappings
        self.node_id_map = {}
        self.node_type_map = {}
        self.edge_type_map = {}
        self.next_node_id = 0
        
        # Statistics
        self.stats = {
            'num_events': 0,
            'num_nodes': 0,
            'num_edges': 0,
            'node_types': defaultdict(int),
            'edge_types': defaultdict(int),
            'time_range': None
        }
    
    def load_json_file(self, file_path: Path, chunk_size: int = 10000):
        """Load large JSON file in chunks."""
        logger.info(f"Loading JSON file: {file_path.name}")
        
        events = []
        
        try:
            # Try to load as JSON array
            with open(file_path, 'r') as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    # JSON array - load in chunks
                    content = f.read()
                    data = json.loads(content)
                    
                    # Process with progress bar
                    with tqdm(total=len(data), desc=f"Processing {file_path.name}", unit="events") as pbar:
                        for i in range(0, len(data), chunk_size):
                            chunk = data[i:i+chunk_size]
                            events.extend(self._process_events(chunk))
                            pbar.update(len(chunk))
                else:
                    # NDJSON - one JSON per line
                    # Count total lines first for progress bar
                    f.seek(0)
                    total_lines = sum(1 for _ in f if _.strip())
                    f.seek(0)
                    
                    with tqdm(total=total_lines, desc=f"Processing {file_path.name}", unit="lines") as pbar:
                        for line in f:
                            if line.strip():
                                try:
                                    event = json.loads(line)
                                    events.append(self._process_event(event))
                                except json.JSONDecodeError:
                                    logger.warning(f"Skipping invalid JSON line")
                            pbar.update(1)
        
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return []
        
        logger.info(f"✓ Loaded {len(events)} events from {file_path.name}")
        return events
    
    def _process_event(self, event: Dict) -> Dict:
        """Process a single event."""
        # Extract fields based on schema
        schema = self.format_config.get('schema', 'elastic')
        
        if schema == 'elastic':
            return self._process_elastic_event(event)
        else:
            return self._process_custom_event(event)
    
    def _process_elastic_event(self, event: Dict) -> Dict:
        """Process Elastic/ELK format event."""
        # Determine event category
        dataset = event.get('data_stream', {}).get('dataset', '')
        event_info = event.get('event', {})
        event_action = event_info.get('action', 'unknown')
        event_category = event_info.get('category', ['unknown'])
        if isinstance(event_category, list):
            event_category = event_category[0] if event_category else 'unknown'
        
        # Extract process information (source)
        process = event.get('process', {})
        source_entity = None
        source_type = 'process'
        
        # Use entity_id as primary identifier for processes
        if 'entity_id' in process:
            source_entity = process['entity_id']
        elif 'executable' in process:
            source_entity = f"{process.get('executable', '')}:{process.get('pid', '')}"
        elif 'name' in process:
            source_entity = f"{process.get('name', '')}:{process.get('pid', '')}"
        
        # Extract target based on event category
        target_entity = None
        target_type = 'unknown'
        
        if event_category == 'file':
            file_info = event.get('file', {})
            target_entity = file_info.get('path', file_info.get('name', ''))
            target_type = 'file'
        elif event_category == 'network':
            # For network events, create a connection identifier
            destination = event.get('destination', {})
            source_net = event.get('source', {})
            dest_ip = destination.get('ip', '')
            dest_port = destination.get('port', '')
            src_ip = source_net.get('ip', '')
            src_port = source_net.get('port', '')
            
            if dest_ip and dest_port:
                target_entity = f"{dest_ip}:{dest_port}"
            elif dest_ip:
                target_entity = dest_ip
            target_type = 'network'
            
            # Store connection details
            if src_ip and src_port:
                source_entity = f"{src_ip}:{src_port}"
        elif event_category == 'process':
            # Process events (fork, exec, exit)
            parent = process.get('parent', {})
            if parent and 'entity_id' in parent:
                target_entity = parent['entity_id']
                target_type = 'process'
            else:
                target_entity = process.get('entity_id', '')
                target_type = 'process'
        
        processed = {
            'timestamp': event.get('@timestamp', ''),
            'source': source_entity or 'unknown',
            'source_type': source_type,
            'target': target_entity or 'unknown',
            'target_type': target_type,
            'event_type': event_action,
            'event_category': event_category,
            'attributes': {
                'process_name': process.get('name', ''),
                'process_executable': process.get('executable', ''),
                'process_pid': process.get('pid', ''),
                'user': event.get('user', {}).get('name', ''),
                'host': event.get('host', {}).get('hostname', '')
            }
        }
        
        # Add category-specific attributes
        if event_category == 'file':
            file_info = event.get('file', {})
            processed['attributes'].update({
                'file_path': file_info.get('path', ''),
                'file_name': file_info.get('name', ''),
                'file_extension': file_info.get('extension', ''),
                'original_path': file_info.get('Ext', {}).get('original', {}).get('path', '')
            })
        elif event_category == 'network':
            destination = event.get('destination', {})
            source_net = event.get('source', {})
            network = event.get('network', {})
            processed['attributes'].update({
                'dest_ip': destination.get('ip', ''),
                'dest_port': destination.get('port', ''),
                'src_ip': source_net.get('ip', ''),
                'src_port': source_net.get('port', ''),
                'transport': network.get('transport', ''),
                'network_type': network.get('type', '')
            })
        
        return processed
    
    def _process_custom_event(self, event: Dict) -> Dict:
        """Process custom format event."""
        event_field = self.format_config.get('event_field', 'timestamp')
        source_field = self.format_config.get('source_field', 'source')
        target_field = self.format_config.get('target_field', 'target')
        type_field = self.format_config.get('type_field', 'type')
        
        return {
            'timestamp': event.get(event_field, ''),
            'source': event.get(source_field, {}),
            'target': event.get(target_field, {}),
            'event_type': event.get(type_field, 'unknown'),
            'attributes': event
        }
    
    def _process_events(self, events: List[Dict]) -> List[Dict]:
        """Process a batch of events."""
        return [self._process_event(e) for e in events]
    
    def get_or_create_node_id(self, entity: str, entity_type: str) -> int:
        """Get or create node ID for an entity."""
        key = f"{entity_type}:{entity}"
        
        if key not in self.node_id_map:
            self.node_id_map[key] = self.next_node_id
            self.node_type_map[self.next_node_id] = entity_type
            self.next_node_id += 1
            self.stats['node_types'][entity_type] += 1
        
        return self.node_id_map[key]
    
    def extract_entity(self, entity_data: Dict, default_type: str) -> Tuple[str, str]:
        """Extract entity identifier and type."""
        if isinstance(entity_data, str):
            return entity_data, default_type
        
        # Try different fields
        entity_id = (
            entity_data.get('name') or
            entity_data.get('path') or
            entity_data.get('ip') or
            entity_data.get('id') or
            entity_data.get('entity_id') or
            str(entity_data)
        )
        
        entity_type = entity_data.get('type', default_type)
        
        return entity_id, entity_type
    
    def build_graph(self, events: List[Dict]) -> Dict:
        """Build graph from events."""
        logger.info(f"\n{'='*80}")
        logger.info(f"  Step 2/4: Building provenance graph")
        logger.info(f"{'='*80}\n")
        
        edges = []
        node_features = defaultdict(list)
        edge_features = defaultdict(list)
        timestamps = []
        
        skipped_events = 0
        
        for event in tqdm(events, desc="Building graph", unit="event", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            # Extract source and target entities
            source_entity = event.get('source')
            target_entity = event.get('target')
            source_type = event.get('source_type', 'process')
            target_type = event.get('target_type', 'file')
            
            # Skip events with missing source or target
            if not source_entity or not target_entity or source_entity == 'unknown' or target_entity == 'unknown':
                skipped_events += 1
                continue
            
            # For string entities, extract identifier and type
            if isinstance(source_entity, dict):
                source_entity, source_type = self.extract_entity(source_entity, source_type)
            if isinstance(target_entity, dict):
                target_entity, target_type = self.extract_entity(target_entity, target_type)
            
            # Create node IDs
            source_id = self.get_or_create_node_id(source_entity, source_type)
            target_id = self.get_or_create_node_id(target_entity, target_type)
            
            # Skip self-loops
            if source_id == target_id:
                skipped_events += 1
                continue
            
            # Edge type
            event_category = event.get('event_category', 'unknown')
            event_action = event.get('event_type', 'unknown')
            
            # Create a more descriptive edge type
            edge_type = f"{event_category}_{event_action}"
            edge_type_id = self.get_or_create_edge_type_id(edge_type)
            
            # Add edge
            edges.append((source_id, target_id, edge_type_id))
            
            # Store node features (attributes from the event)
            attributes = event.get('attributes', {})
            if attributes:
                node_features[source_id].append(attributes)
                node_features[target_id].append(attributes)
            
            # Timestamp
            timestamp = self._parse_timestamp(event.get('timestamp'))
            timestamps.append(timestamp)
            
            # Update statistics
            self.stats['num_events'] += 1
            self.stats['edge_types'][edge_type] += 1
        
        self.stats['num_nodes'] = self.next_node_id
        self.stats['num_edges'] = len(edges)
        
        logger.info(f"\n✓ Created {self.next_node_id:,} nodes")
        logger.info(f"✓ Created {len(edges):,} edges")
        if skipped_events > 0:
            logger.info(f"⚠ Skipped {skipped_events:,} events (missing entities or self-loops)\n")
        
        if timestamps:
            self.stats['time_range'] = (min(timestamps), max(timestamps))
        
        graph_data = {
            'edges': edges,
            'num_nodes': self.next_node_id,
            'node_id_map': self.node_id_map,
            'node_type_map': self.node_type_map,
            'edge_type_map': self.edge_type_map,
            'node_features': dict(node_features),
            'timestamps': timestamps,
            'stats': self.stats
        }
        
        return graph_data
    
    def get_or_create_edge_type_id(self, edge_type: str) -> int:
        """Get or create edge type ID."""
        if edge_type not in self.edge_type_map:
            self.edge_type_map[edge_type] = len(self.edge_type_map)
        return self.edge_type_map[edge_type]
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string to Unix timestamp."""
        if not timestamp_str:
            return 0.0
        
        try:
            # Try ISO format
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except:
            try:
                # Try as Unix timestamp
                return float(timestamp_str)
            except:
                return 0.0
    
    def save_graph(self, graph_data: Dict, output_path: Path):
        """Save graph data."""
        logger.info(f"\n{'='*80}")
        logger.info(f"  Step 3/4: Saving graph data")
        logger.info(f"{'='*80}\n")
        logger.info(f"Output path: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        logger.info("Saving graph pickle...")
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
        logger.info(f"✓ Graph saved: {output_path}")
        
        # Save statistics as JSON
        logger.info("Saving statistics...")
        stats_path = output_path.with_suffix('.json')
        with open(stats_path, 'w') as f:
            # Convert defaultdict to dict for JSON serialization
            stats_json = {
                'num_events': self.stats['num_events'],
                'num_nodes': self.stats['num_nodes'],
                'num_edges': self.stats['num_edges'],
                'node_types': dict(self.stats['node_types']),
                'edge_types': dict(self.stats['edge_types']),
                'time_range': self.stats['time_range']
            }
            json.dump(stats_json, f, indent=2)
        
        logger.info(f"✓ Statistics saved: {stats_path}\n")
    
    def print_statistics(self):
        """Print preprocessing statistics."""
        print("\n" + "="*80)
        print("Preprocessing Statistics")
        print("="*80)
        print(f"Total events: {self.stats['num_events']:,}")
        print(f"Total nodes: {self.stats['num_nodes']:,}")
        print(f"Total edges: {self.stats['num_edges']:,}")
        print(f"\nNode types:")
        for node_type, count in self.stats['node_types'].items():
            print(f"  {node_type:15s}: {count:,}")
        print(f"\nEdge types:")
        for edge_type, count in self.stats['edge_types'].items():
            print(f"  {edge_type:15s}: {count:,}")
        
        if self.stats['time_range']:
            start, end = self.stats['time_range']
            print(f"\nTime range:")
            print(f"  Start: {datetime.fromtimestamp(start)}")
            print(f"  End:   {datetime.fromtimestamp(end)}")
            print(f"  Duration: {(end - start) / 3600:.2f} hours")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess custom SOC data for PIDS models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=project_root / 'configs' / 'datasets' / 'custom_soc.yaml',
        help='Dataset configuration file'
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing JSON files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / 'data' / 'processed',
        help='Output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='custom_soc',
        help='Name of the dataset (used for output filename)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for processing large files'
    )
    
    args = parser.parse_args()
    
    # Construct output path from output-dir and dataset-name
    output_file = args.output_dir / f'{args.dataset_name}_graph.pkl'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override input directory if specified
    if args.input_dir:
        config['data']['root_dir'] = str(args.input_dir)
    
    # Create preprocessor
    preprocessor = SOCDataPreprocessor(config)
    
    # Load JSON files
    input_dir = Path(config['data']['root_dir'])
    json_files = config['data'].get('files', [])
    
    if not json_files:
        # Find all JSON files in directory
        json_files = list(input_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    else:
        json_files = [input_dir / f for f in json_files]
    
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return
    
    # Process all files with progress bar
    logger.info(f"\n{'='*80}")
    logger.info(f"  Step 1/4: Loading and processing JSON files")
    logger.info(f"{'='*80}\n")
    
    all_events = []
    for json_file in tqdm(json_files, desc="Loading files", unit="file"):
        if not json_file.exists():
            logger.warning(f"File not found: {json_file}")
            continue
        
        events = preprocessor.load_json_file(json_file, args.chunk_size)
        all_events.extend(events)
    
    if not all_events:
        logger.error("No events loaded!")
        return
    
    logger.info(f"\n✓ Total events loaded: {len(all_events):,}\n")
    
    # Build graph
    graph_data = preprocessor.build_graph(all_events)
    
    # Save graph
    preprocessor.save_graph(graph_data, output_file)
    
    # Print statistics
    logger.info(f"{'='*80}")
    logger.info(f"  Step 4/4: Final Statistics")
    logger.info(f"{'='*80}\n")
    preprocessor.print_statistics()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"  ✓ Preprocessing Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Output: {output_file}\n")
    logger.info(f"Statistics saved to: {output_file.with_suffix('.json')}")


if __name__ == '__main__':
    main()
