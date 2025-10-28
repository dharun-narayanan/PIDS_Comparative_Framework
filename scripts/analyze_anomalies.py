#!/usr/bin/env python3
"""
Anomaly Detection Analysis Script
Extracts and analyzes top anomalies from PIDS model predictions
"""

import pickle
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple


def load_model_scores(artifacts_dir: Path, model: str) -> np.ndarray:
    """Load anomaly scores from model inference output"""
    output_path = artifacts_dir / model / 'model_inference' / 'output.pkl'
    with open(output_path, 'rb') as f:
        output = pickle.load(f)
    return np.array(output['scores'])


def extract_top_k_anomalies(scores: np.ndarray, k: int = 100) -> List[Tuple[int, float]]:
    """Extract top K anomalous events by score"""
    top_indices = scores.argsort()[-k:][::-1]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def calculate_statistics(scores: np.ndarray) -> Dict:
    """Calculate statistical measures for anomaly scores"""
    return {
        'count': len(scores),
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'min': float(scores.min()),
        'max': float(scores.max()),
        'percentiles': {
            '50': float(np.percentile(scores, 50)),
            '90': float(np.percentile(scores, 90)),
            '95': float(np.percentile(scores, 95)),
            '99': float(np.percentile(scores, 99)),
            '99.5': float(np.percentile(scores, 99.5)),
            '99.9': float(np.percentile(scores, 99.9)),
        }
    }


def load_graph(data_path: Path, dataset: str):
    """Load the provenance graph"""
    graph_path = data_path / f'{dataset}_graph.pkl'
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def get_event_details(graph, event_id: int) -> Dict:
    """Extract detailed information about a specific event"""
    # Handle both NetworkX graph and dict formats
    if hasattr(graph, 'edges'):
        edges = list(graph.edges(data=True))
        
        if event_id >= len(edges):
            return {'error': 'Event ID out of range'}
        
        src, dst, data = edges[event_id]
        
        return {
            'event_id': event_id,
            'event_type': data.get('event_type', 'unknown'),
            'timestamp': data.get('timestamp', 0),
            'datetime': datetime.fromtimestamp(data.get('timestamp', 0)).isoformat() if data.get('timestamp', 0) > 0 else None,
            'source': {
                'node_id': src,
                'node_type': graph.nodes[src].get('node_type', 'unknown'),
                'attributes': dict(graph.nodes[src])
            },
            'target': {
                'node_id': dst,
                'node_type': graph.nodes[dst].get('node_type', 'unknown'),
                'attributes': dict(graph.nodes[dst])
            },
            'edge_attributes': dict(data)
        }
    else:
        # Handle dict format (preprocessed graph data)
        return {
            'event_id': event_id,
            'note': 'Detailed event information requires NetworkX graph format'
        }


def ensemble_detection(artifacts_dir: Path, models: List[str], threshold_percentile: float = 99) -> np.ndarray:
    """Perform ensemble anomaly detection by averaging normalized scores"""
    all_scores = []
    
    for model in models:
        scores = load_model_scores(artifacts_dir, model)
        # Normalize to [0, 1]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        all_scores.append(scores_norm)
    
    # Average across models
    ensemble_scores = np.mean(all_scores, axis=0)
    return ensemble_scores


def temporal_analysis(graph, scores: np.ndarray, time_window_hours: int = 1) -> List[Dict]:
    """Analyze anomaly scores over time windows"""
    # Handle both NetworkX graph and dict formats
    if hasattr(graph, 'edges'):
        edges = list(graph.edges(data=True))
        
        # Group by time window
        time_buckets = defaultdict(list)
        for idx, (_, _, data) in enumerate(edges):
            ts = data.get('timestamp', 0)
            if ts > 0:
                bucket = int(ts) // (time_window_hours * 3600) * (time_window_hours * 3600)
                time_buckets[bucket].append((idx, scores[idx]))
        
        # Calculate statistics per bucket
        results = []
        for bucket_ts, events in sorted(time_buckets.items()):
            event_scores = [score for _, score in events]
            results.append({
                'timestamp': bucket_ts,
                'datetime': datetime.fromtimestamp(bucket_ts).isoformat(),
                'num_events': len(events),
                'avg_score': float(np.mean(event_scores)),
                'max_score': float(np.max(event_scores)),
                'std_score': float(np.std(event_scores)),
                'high_score_count': sum(1 for s in event_scores if s > np.percentile(scores, 99))
            })
        
        # Sort by average score (descending)
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        return results
    else:
        return [{'note': 'Temporal analysis requires NetworkX graph format'}]


def main():
    parser = argparse.ArgumentParser(description='Analyze anomaly detection results')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts',
                        help='Path to artifacts directory')
    parser.add_argument('--data-path', type=str, default='data/custom_soc',
                        help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='custom_soc',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='magic',
                        choices=['magic', 'kairos', 'continuum_fl', 'orthrus', 'threatrace'],
                        help='Model to analyze')
    parser.add_argument('--output-dir', type=str, default='results/anomaly_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of top anomalies to extract')
    parser.add_argument('--with-details', action='store_true',
                        help='Include detailed event information (slower)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Perform ensemble detection across all models')
    parser.add_argument('--temporal', action='store_true',
                        help='Perform temporal analysis')
    
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PIDS Anomaly Detection Analysis")
    print("=" * 80)
    
    # Load model scores
    if args.ensemble:
        print("\n📊 Performing ensemble detection...")
        models = ['magic', 'continuum_fl', 'threatrace']
        scores = ensemble_detection(artifacts_dir, models)
        model_name = 'ensemble'
    else:
        print(f"\n📊 Analyzing model: {args.model}")
        scores = load_model_scores(artifacts_dir, args.model)
        model_name = args.model
    
    # Calculate statistics
    print("\n📈 Computing statistics...")
    stats = calculate_statistics(scores)
    
    print(f"\nScore Statistics:")
    print(f"  Events: {stats['count']:,}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std Dev: {stats['std']:.6f}")
    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"\nPercentile Thresholds:")
    for p, v in stats['percentiles'].items():
        count = np.sum(scores > v)
        print(f"  {p}th: {v:.6f} ({count:,} events above)")
    
    # Extract top anomalies
    print(f"\n🔍 Extracting top {args.top_k} anomalies...")
    top_anomalies = extract_top_k_anomalies(scores, k=args.top_k)
    
    # Save top anomalies
    anomaly_data = {
        'model': model_name,
        'dataset': args.dataset,
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'top_anomalies': [
            {
                'rank': i + 1,
                'event_id': event_id,
                'anomaly_score': score
            }
            for i, (event_id, score) in enumerate(top_anomalies)
        ]
    }
    
    # Add detailed event information if requested
    if args.with_details:
        print("\n📝 Loading event details...")
        graph = load_graph(data_path, args.dataset)
        
        for item in anomaly_data['top_anomalies']:
            event_id = item['event_id']
            details = get_event_details(graph, event_id)
            item['details'] = details
    
    # Save results
    output_file = output_dir / f'{model_name}_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(anomaly_data, f, indent=2)
    print(f"\n✅ Saved analysis to: {output_file}")
    
    # Temporal analysis
    if args.temporal:
        print("\n⏰ Performing temporal analysis...")
        graph = load_graph(data_path, args.dataset)
        temporal_results = temporal_analysis(graph, scores, time_window_hours=1)
        
        temporal_file = output_dir / f'{model_name}_temporal.json'
        with open(temporal_file, 'w') as f:
            json.dump(temporal_results[:50], f, indent=2)  # Top 50 suspicious time windows
        
        print(f"\nTop 5 Suspicious Time Windows:")
        for i, window in enumerate(temporal_results[:5], 1):
            print(f"  {i}. {window['datetime']}")
            print(f"     Events: {window['num_events']:,}, Avg Score: {window['avg_score']:.6f}")
            print(f"     High-score events: {window['high_score_count']}")
        
        print(f"\n✅ Saved temporal analysis to: {temporal_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Total Events: {stats['count']:,}")
    print(f"Top {args.top_k} Anomalies Extracted")
    print(f"\nRecommended Thresholds:")
    print(f"  🔴 Critical (99.9%): > {stats['percentiles']['99.9']:.6f} ({np.sum(scores > stats['percentiles']['99.9']):,} events)")
    print(f"  🟡 High (99%):      > {stats['percentiles']['99']:.6f} ({np.sum(scores > stats['percentiles']['99']):,} events)")
    print(f"  🔵 Medium (95%):    > {stats['percentiles']['95']:.6f} ({np.sum(scores > stats['percentiles']['95']):,} events)")
    print("\nNext steps:")
    print("  1. Review top anomalies in the JSON output")
    print("  2. Investigate high-score events manually")
    print("  3. Create ground truth labels for confirmed threats")
    print("  4. Retrain models with labeled data")
    print("=" * 80)


if __name__ == '__main__':
    main()
