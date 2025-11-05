#!/usr/bin/env python3
"""
Universal Preprocessor for Provenance Data

This unified script handles preprocessing for ALL data formats:
- Custom SOC logs (Elastic/ELK, NDJSON, JSON arrays)
- DARPA TC datasets (CDM v18 - JSON NDJSON and binary AVRO)
- Any custom JSON format with configurable schema

The script automatically detects the format and processes accordingly.
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.semantic_parser import SemanticParser, Event, Entity
from utils.common import setup_logging, load_config
import logging

logger = setup_logging()
logger = logging.getLogger("preprocess")


class UniversalPreprocessor:
    """Universal preprocessor for all provenance data formats."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get('data', {})
        self.format_config = config.get('format', {})
        self.graph_config = config.get('graph', {})
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Initialize semantic parser
        parser_config = {
            'schema': self.format_config,
            'graph': self.graph_config
        }
        self.parser = SemanticParser(parser_config)
        
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
            'time_range': None,
            'files_processed': 0,
            'parse_errors': 0,
            'format_detected': None,
            'dataset_type': config.get('dataset_type', 'unknown')
        }
    
    def load_and_parse_files(self, file_paths: List[Path], 
                            max_events_per_file: Optional[int] = None) -> List[Event]:
        """
        Load and parse multiple provenance data files using semantic parser.
        
        Args:
            file_paths: List of file paths to process
            max_events_per_file: Maximum events to parse per file (None = all)
            
        Returns:
            List of parsed Event objects
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"  Step 1/4: Loading and parsing provenance data")
        logger.info(f"{'='*80}\n")
        logger.info(f"Dataset type: {self.stats['dataset_type']}")
        logger.info(f"Files to process: {len(file_paths)}")
        
        all_events = []
        
        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                logger.info(f"\nParsing: {file_path.name}")
                events = self.parser.parse_file(file_path, max_events_per_file)
                all_events.extend(events)
                self.stats['files_processed'] += 1
                
                logger.info(f"✓ Extracted {len(events)} events from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.stats['parse_errors'] += 1
                continue
        
        # Get parser statistics
        parser_stats = self.parser.get_statistics()
        self.stats['format_detected'] = self.parser.selected_parser.__class__.__name__ if self.parser.selected_parser else "Unknown"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"  Parsing Statistics")
        logger.info(f"{'='*80}")
        logger.info(f"  Format detected: {self.stats['format_detected']}")
        
        # Show record type breakdown for DARPA datasets
        record_types = {k: v for k, v in parser_stats.items() if k.startswith('record_type_')}
        if record_types:
            logger.info(f"\n  CDM Record Type Breakdown:")
            for record_type, count in sorted(record_types.items(), key=lambda x: x[1], reverse=True):
                type_name = record_type.replace('record_type_', '')
                logger.info(f"    {type_name}: {count:,}")
        
        # Show other statistics
        for key, value in parser_stats.items():
            if not isinstance(value, dict) and not key.startswith('record_type_'):
                logger.info(f"  {key}: {value}")
        
        logger.info(f"\n✓ Total events extracted: {len(all_events):,}\n")
        
        return all_events
    
    def get_or_create_node_id(self, entity: Entity) -> int:
        """Get or create node ID for an entity."""
        key = f"{entity.entity_type}:{entity.entity_id}"
        
        if key not in self.node_id_map:
            self.node_id_map[key] = self.next_node_id
            self.node_type_map[self.next_node_id] = entity.entity_type
            self.next_node_id += 1
            self.stats['node_types'][entity.entity_type] += 1
        
        return self.node_id_map[key]
    
    def get_or_create_edge_type_id(self, edge_type: str) -> int:
        """Get or create edge type ID."""
        if edge_type not in self.edge_type_map:
            self.edge_type_map[edge_type] = len(self.edge_type_map)
        return self.edge_type_map[edge_type]
    
    def build_graph(self, events: List[Event]) -> Dict:
        """Build provenance graph from parsed events."""
        logger.info(f"\n{'='*80}")
        logger.info(f"  Step 2/4: Building provenance graph")
        logger.info(f"{'='*80}\n")
        
        edges = []
        node_features = defaultdict(list)
        edge_features = []
        timestamps = []
        
        skipped_events = 0
        
        # Sort events by timestamp for temporal consistency
        events.sort(key=lambda e: e.timestamp)
        
        for event in tqdm(events, desc="Building graph", unit="event",
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            
            # Skip events with missing entities
            if not event.source or not event.target:
                skipped_events += 1
                continue
            
            # Create node IDs
            source_id = self.get_or_create_node_id(event.source)
            target_id = self.get_or_create_node_id(event.target)
            
            # Skip self-loops if configured
            if self.preprocessing_config.get('remove_self_loops', True):
                if source_id == target_id:
                    skipped_events += 1
                    continue
            
            # Get edge type ID
            edge_type_id = self.get_or_create_edge_type_id(event.event_type)
            
            # Add edge with features
            edges.append((source_id, target_id, edge_type_id))
            
            # Store edge features
            edge_features.append({
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'attributes': event.attributes
            })
            
            # Store node features
            if event.source.attributes:
                node_features[source_id].append(event.source.attributes)
            if event.target.attributes:
                node_features[target_id].append(event.target.attributes)
            
            # Timestamp
            timestamps.append(event.timestamp)
            
            # Update statistics
            self.stats['num_events'] += 1
            self.stats['edge_types'][event.event_type] += 1
        
        self.stats['num_nodes'] = self.next_node_id
        self.stats['num_edges'] = len(edges)
        
        logger.info(f"\n✓ Created {self.next_node_id:,} nodes")
        logger.info(f"✓ Created {len(edges):,} edges")
        if skipped_events > 0:
            logger.info(f"⚠ Skipped {skipped_events:,} events (missing entities or self-loops)\n")
        
        # Calculate time range
        if timestamps:
            self.stats['time_range'] = (min(timestamps), max(timestamps))
        
        # Build reverse mapping (node_id -> entity)
        node_id_to_entity = {}
        entity_registry = self.parser.get_entity_registry()
        for entity_key, entity in entity_registry.items():
            node_id = self.node_id_map.get(entity_key)
            if node_id is not None:
                node_id_to_entity[node_id] = {
                    'entity_id': entity.entity_id,
                    'entity_type': entity.entity_type,
                    'attributes': entity.attributes
                }
        
        graph_data = {
            'edges': edges,
            'num_nodes': self.next_node_id,
            'node_id_map': self.node_id_map,
            'node_type_map': self.node_type_map,
            'edge_type_map': self.edge_type_map,
            'node_features': dict(node_features),
            'edge_features': edge_features,
            'timestamps': timestamps,
            'node_id_to_entity': node_id_to_entity,
            'stats': self.stats
        }
        
        return graph_data
    
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
            pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Graph saved: {output_path} ({file_size_mb:.2f} MB)")
        
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
                'time_range': self.stats['time_range'],
                'files_processed': self.stats['files_processed'],
                'parse_errors': self.stats['parse_errors'],
                'format_detected': self.stats['format_detected'],
                'dataset_type': self.stats['dataset_type']
            }
            json.dump(stats_json, f, indent=2)
        
        logger.info(f"✓ Statistics saved: {stats_path}\n")
    
    def print_statistics(self):
        """Print preprocessing statistics."""
        print("\n" + "="*80)
        print("Preprocessing Statistics")
        print("="*80)
        print(f"Dataset type: {self.stats['dataset_type']}")
        print(f"Format detected: {self.stats['format_detected']}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Parse errors: {self.stats['parse_errors']}")
        print(f"Total events: {self.stats['num_events']:,}")
        print(f"Total nodes: {self.stats['num_nodes']:,}")
        print(f"Total edges: {self.stats['num_edges']:,}")
        print(f"\nNode types:")
        for node_type, count in sorted(self.stats['node_types'].items(), 
                                      key=lambda x: x[1], reverse=True):
            percentage = (count / self.stats['num_nodes'] * 100) if self.stats['num_nodes'] > 0 else 0
            print(f"  {node_type:15s}: {count:8,} ({percentage:5.2f}%)")
        print(f"\nEdge types (top 20):")
        for edge_type, count in sorted(self.stats['edge_types'].items(), 
                                      key=lambda x: x[1], reverse=True)[:20]:
            percentage = (count / self.stats['num_events'] * 100) if self.stats['num_events'] > 0 else 0
            print(f"  {edge_type:20s}: {count:8,} ({percentage:5.2f}%)")
        
        if self.stats['edge_types'] and len(self.stats['edge_types']) > 20:
            print(f"  ... and {len(self.stats['edge_types']) - 20} more edge types")
        
        if self.stats['time_range']:
            start, end = self.stats['time_range']
            duration_hours = (end - start) / 3600
            duration_days = duration_hours / 24
            print(f"\nTime range:")
            print(f"  Start:    {datetime.fromtimestamp(start)}")
            print(f"  End:      {datetime.fromtimestamp(end)}")
            print(f"  Duration: {duration_days:.2f} days ({duration_hours:.2f} hours)")
        print("="*80 + "\n")


def auto_detect_dataset_type(input_path: Path) -> str:
    """Auto-detect dataset type from path or file structure."""
    path_str = str(input_path).lower()
    
    # DARPA detection
    if 'darpa' in path_str or 'ta1-' in path_str:
        return 'darpa'
    
    # Check for DARPA dataset names
    darpa_datasets = ['cadets', 'theia', 'trace', 'clearscope']
    for ds in darpa_datasets:
        if ds in path_str:
            return 'darpa'
    
    # Custom SOC detection
    if 'soc' in path_str or 'elastic' in path_str or 'endpoint' in path_str:
        return 'custom_soc'
    
    # Default to custom
    return 'custom'


def main():
    parser = argparse.ArgumentParser(
        description="Universal preprocessor for provenance data (Custom SOC, DARPA TC, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess custom SOC data
  python scripts/preprocess.py \\
    --input-dir data/custom_soc \\
    --dataset-name custom_soc \\
    --dataset-type custom_soc

  # Preprocess DARPA CADETS dataset
  python scripts/preprocess.py \\
    --input-dir ../DARPA/ta1-cadets-e3-official-1.json \\
    --dataset-name cadets_e3 \\
    --dataset-type darpa

  # Auto-detect and preprocess with sampling
  python scripts/preprocess.py \\
    --input-dir ../DARPA/ta1-theia-e3-official-1r.json \\
    --dataset-name theia_e3 \\
    --max-events-per-file 100000

  # Use custom configuration
  python scripts/preprocess.py \\
    --input-dir ../DARPA/ta1-trace-e3-official-1.json \\
    --dataset-name trace_e3 \\
    --config configs/datasets/trace_e3.yaml
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        help='Input directory containing provenance data files'
    )
    
    parser.add_argument(
        '--input-files',
        type=Path,
        nargs='+',
        help='Specific input files to process (alternative to --input-dir)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / 'data',
        help='Output directory for preprocessed data (default: data/)'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Name of the dataset (e.g., cadets_e3, custom_soc, my_dataset)'
    )
    
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['darpa', 'custom_soc', 'custom'],
        help='Type of dataset (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Dataset configuration file (optional, auto-selected if not provided)'
    )
    
    parser.add_argument(
        '--max-events-per-file',
        type=int,
        help='Maximum number of events to parse per file (for testing/sampling)'
    )
    
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.json',
        help='File pattern to match when using --input-dir (default: *.json)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'ndjson', 'bin', 'avro', 'auto'],
        default='auto',
        help='Force specific file format (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    input_files = []
    
    if args.input_files:
        input_files = args.input_files
        logger.info(f"Processing {len(input_files)} specified files")
    elif args.input_dir:
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        
        # Auto-adjust file pattern based on format if using default pattern
        file_pattern = args.file_pattern
        if file_pattern == '*.json' and args.format in ['bin', 'avro']:
            # For binary AVRO, match both .bin and .bin.* files
            logger.info(f"Auto-adjusted file pattern to '*.bin*' for format '{args.format}'")
            # Find all files matching .bin or .bin.*
            input_files = sorted(args.input_dir.glob('*.bin'))
            input_files.extend(sorted(args.input_dir.glob('*.bin.*')))
            input_files = sorted(set(input_files))  # Remove duplicates and sort
            logger.info(f"Found {len(input_files)} files matching '*.bin*' in {args.input_dir}")
        else:
            # Find all matching files
            input_files = sorted(args.input_dir.glob(file_pattern))
            logger.info(f"Found {len(input_files)} files matching '{file_pattern}' in {args.input_dir}")
    else:
        logger.error("Either --input-dir or --input-files must be specified")
        parser.print_help()
        return 1
    
    if not input_files:
        logger.error("No input files found")
        return 1
    
    # Auto-detect dataset type if not specified
    if not args.dataset_type:
        args.dataset_type = auto_detect_dataset_type(args.input_dir if args.input_dir else input_files[0].parent)
        logger.info(f"Auto-detected dataset type: {args.dataset_type}")
    
    # Determine output subdirectory based on dataset type
    if args.dataset_type == 'darpa':
        output_subdir = 'darpa'
    elif args.dataset_type == 'custom_soc':
        output_subdir = 'custom_soc'
    else:
        output_subdir = 'processed'
    
    output_file = args.output_dir / output_subdir / f'{args.dataset_name}_graph.pkl'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Log file information
    total_size = sum(f.stat().st_size for f in input_files if f.exists()) / (1024**3)  # GB
    logger.info(f"Total input size: {total_size:.2f} GB")
    
    # Load or auto-select configuration
    if args.config and args.config.exists():
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        # Try to auto-select config based on dataset name
        config_path = project_root / 'configs' / 'datasets' / f'{args.dataset_name}.yaml'
        if config_path.exists():
            logger.info(f"Auto-selected configuration: {config_path}")
            config = load_config(config_path)
        else:
            # Use default configuration based on dataset type
            logger.info(f"Using default {args.dataset_type} configuration")
            if args.dataset_type == 'darpa':
                config = {
                    'data': {
                        'dataset_name': args.dataset_name,
                        'format': 'darpa_cdm'
                    },
                    'format': {
                        'type': 'darpa_cdm',
                        'version': '18'
                    },
                    'graph': {
                        'node_types': ['process', 'file', 'network', 'memory', 'socket', 'registry'],
                        'temporal': {
                            'enabled': True,
                            'window_size': 3600
                        }
                    },
                    'preprocessing': {
                        'remove_duplicates': False,
                        'remove_self_loops': True,
                        'entity_resolution': {
                            'enabled': True
                        }
                    },
                    'dataset_type': 'darpa'
                }
            else:  # custom_soc or custom
                config = {
                    'data': {
                        'dataset_name': args.dataset_name
                    },
                    'format': {
                        'type': 'elastic',
                        'schema': 'elastic'
                    },
                    'graph': {
                        'node_types': ['process', 'file', 'network'],
                        'temporal': {
                            'enabled': True,
                            'window_size': 3600
                        }
                    },
                    'preprocessing': {
                        'remove_self_loops': True
                    },
                    'dataset_type': args.dataset_type
                }
    
    # Ensure dataset_type is in config
    config['dataset_type'] = args.dataset_type
    
    # Create preprocessor
    preprocessor = UniversalPreprocessor(config)
    
    # Load and parse files
    events = preprocessor.load_and_parse_files(
        input_files, 
        max_events_per_file=args.max_events_per_file
    )
    
    if not events:
        logger.error("No events were parsed!")
        return 1
    
    # Build graph
    graph_data = preprocessor.build_graph(events)
    
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
    logger.info(f"Statistics: {output_file.with_suffix('.json')}")
    logger.info(f"\nYou can now use this dataset with the evaluation pipeline:")
    logger.info(f"  python experiments/evaluate_pipeline.py \\")
    logger.info(f"    --models magic,kairos,orthrus \\")
    logger.info(f"    --data-path {output_file.parent} \\")
    logger.info(f"    --dataset {args.dataset_name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
