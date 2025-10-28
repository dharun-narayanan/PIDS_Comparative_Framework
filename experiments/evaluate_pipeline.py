#!/usr/bin/env python3
"""
Task-based evaluation script for PIDS models.

This script uses the modular pipeline architecture to evaluate models
on preprocessed datasets using pretrained weights.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.pipeline_builder import PipelineBuilder
from utils.common import setup_logging, set_seed

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Deep merge two configuration dictionaries."""
    result = base_config.copy()
    
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(result, override_config)
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate PIDS models using task-based pipeline'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated list of models to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='custom_soc',
        help='Dataset name (default: custom_soc)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to preprocessed dataset'
    )
    
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default='checkpoints',
        help='Directory containing model checkpoints'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--artifact-dir',
        type=str,
        default='artifacts',
        help='Directory for intermediate artifacts'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--force-restart',
        action='store_true',
        help='Force re-execution of all tasks (ignore cache)'
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default=None,
        help='Comma-separated list of tasks to run (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration YAML file'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def get_available_models() -> List[str]:
    """Get list of available models from configs/models directory."""
    from models.model_builder import list_models
    return list_models()


def build_global_config(args) -> Dict[str, Any]:
    """
    Build global configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        'data': {
            'path': args.data_path,
            'dataset': args.dataset,
        },
        'checkpoints_dir': args.checkpoints_dir,
        'output_dir': args.output_dir,
        'artifact_dir': args.artifact_dir,
        'device': args.device,
        'seed': args.seed,
        'force_restart': args.force_restart,
    }
    
    # Load custom config if provided
    if args.config:
        custom_config = load_config(Path(args.config))
        config = merge_configs(config, custom_config)
    
    return config


def evaluate_model(
    model_name: str,
    config: Dict[str, Any],
    tasks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate a single model using the task pipeline.
    
    Args:
        model_name: Name of model to evaluate
        config: Global configuration
        tasks: List of tasks to execute (None = all)
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("="*80)
    logger.info(f"Evaluating model: {model_name}")
    logger.info("="*80)
    
    try:
        # Build pipeline
        builder = PipelineBuilder(config)
        
        # Execute pipeline
        results = builder.build_and_execute(
            model_name=model_name,
            tasks=tasks,
            force_restart=config.get('force_restart', False)
        )
        
        # Extract metrics
        metrics = {}
        if 'calculate_metrics' in results:
            metrics = results['calculate_metrics']
        
        logger.info(f"\n{model_name} Results:")
        if metrics:
            # Display unsupervised anomaly detection metrics
            if 'anomaly_score_stats' in metrics:
                stats = metrics['anomaly_score_stats']
                logger.info(f"  Anomaly Score Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                logger.info(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                logger.info(f"  Separation Ratio: {metrics.get('score_separation_ratio', 0):.4f}")
                logger.info(f"  Critical Anomalies (99.9%): {metrics['anomaly_counts']['critical_99.9']}")
                logger.info(f"  High Anomalies (99%): {metrics['anomaly_counts']['high_99']}")
            
            # If supervised metrics available, show them too
            if metrics.get('supervised_metrics'):
                sup = metrics['supervised_metrics']
                logger.info(f"  [Supervised] AUROC: {sup.get('auroc', 0):.4f}, F1: {sup.get('f1_score', 0):.4f}")
        else:
            logger.warning("No metrics available")
        
        return {
            'model': model_name,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'error': str(e),
            'success': False
        }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("="*80)
    logger.info("PIDS Task-Based Evaluation Pipeline")
    logger.info("="*80)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Build configuration
    config = build_global_config(args)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Determine models to evaluate
    if args.models == 'all':
        models = get_available_models()
    else:
        models = [m.strip() for m in args.models.split(',')]
    
    logger.info(f"Models to evaluate: {models}")
    
    # Determine tasks to run
    tasks = None
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
        logger.info(f"Tasks to run: {tasks}")
    
    # Evaluate each model
    all_results = []
    for model_name in models:
        result = evaluate_model(model_name, config, tasks)
        all_results.append(result)
    
    # Save consolidated results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'evaluation_results_{args.dataset}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Summary - Unsupervised Anomaly Detection")
    logger.info("="*80)
    
    successful = sum(1 for r in all_results if r['success'])
    logger.info(f"Models evaluated: {successful}/{len(all_results)}")
    
    # Sort by score separation ratio (best anomaly detector first)
    results_with_scores = []
    for result in all_results:
        if result['success']:
            metrics = result.get('metrics', {})
            sep_ratio = metrics.get('score_separation_ratio', 0)
            results_with_scores.append((result['model'], sep_ratio, metrics))
    
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n{'Model':<20} {'Sep. Ratio':<12} {'Critical':<10} {'High':<10}")
    logger.info("-" * 52)
    for model_name, sep_ratio, metrics in results_with_scores:
        critical = metrics.get('anomaly_counts', {}).get('critical_99.9', 0)
        high = metrics.get('anomaly_counts', {}).get('high_99', 0)
        logger.info(f"{model_name:<20} {sep_ratio:<12.4f} {critical:<10} {high:<10}")
    
    # Show failed models
    for result in all_results:
        if not result['success']:
            logger.info(f"  {result['model']}: FAILED - {result.get('error', 'Unknown error')}")
    
    logger.info("="*80)
    logger.info("Higher separation ratio = better anomaly detection capability")
    logger.info("="*80)


if __name__ == '__main__':
    main()
