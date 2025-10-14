"""
Evaluation script for PIDS models.

Usage:
    python experiments/evaluate.py --model magic --dataset custom --pretrained
    python experiments/evaluate.py --all-models --dataset cadets-e3 --pretrained
"""

import argparse
import sys
from pathlib import Path
import logging
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models import ModelRegistry
from data.dataset import get_dataset
from utils.common import (
    set_seed, setup_logging, load_config,
    get_device, ensure_dir, format_time
)
from utils.metrics import compute_detection_metrics, compute_entity_level_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate PIDS models')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=None,
                       choices=['magic', 'kairos', 'orthrus', 'threatrace', 'continuum_fl'],
                       help='Model to evaluate')
    parser.add_argument('--all-models', action='store_true',
                       help='Evaluate all available models')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (custom, cadets-e3, theia-e3, etc.)')
    parser.add_argument('--data-path', type=Path, default=Path('data/custom'),
                       help='Path to dataset')
    
    # Model loading
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--checkpoint', type=Path, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'),
                       help='Directory containing checkpoints')
    
    # Evaluation options
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--detection-level', type=str, default='entity',
                       choices=['entity', 'batch', 'both'],
                       help='Detection level')
    parser.add_argument('--k-neighbors', type=int, default=5,
                       help='Number of neighbors for k-NN (entity level)')
    
    # System arguments
    parser.add_argument('--device', type=int, default=-1,
                       help='Device: -1 for CPU (default), 0+ for GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help='Directory to save results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to file')
    
    return parser.parse_args()


def evaluate_model(model, dataloader, device, detection_level='entity', k_neighbors=5):
    """
    Evaluate model on dataset.
    
    Args:
        model: PIDS model
        dataloader: Data loader
        device: Device to use
        detection_level: 'entity' or 'batch'
        k_neighbors: Number of neighbors for k-NN
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.set_eval_mode()
    
    all_labels = []
    all_scores = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Extract labels (format depends on data)
                # labels = batch.labels if hasattr(batch, 'labels') else batch[1]
                
                # Get model predictions
                # output = model(batch)
                # scores = output['scores'] if isinstance(output, dict) else output
                
                # Placeholder for actual implementation
                # all_labels.extend(labels.cpu().numpy())
                # all_scores.extend(scores.cpu().numpy())
                
                pass
            except Exception as e:
                logging.error(f'Error in batch {batch_idx}: {e}')
                continue
    
    # For now, return dummy metrics
    # In actual implementation, compute based on all_labels and all_scores
    
    if detection_level == 'entity' or detection_level == 'both':
        # Entity-level metrics
        # metrics = compute_entity_level_metrics(
        #     np.array(all_labels), 
        #     np.array(all_scores),
        #     k_neighbors=k_neighbors
        # )
        metrics = {
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'fpr': 0.0
        }
    else:
        # Batch-level metrics
        metrics = {
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    return metrics, all_predictions


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    ensure_dir(args.output_dir)
    logger = setup_logging(args.output_dir)
    
    logger.info('='*50)
    logger.info('PIDS Model Evaluation')
    logger.info('='*50)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    logger.info(f'Device: {device}')
    
    # Determine models to evaluate
    if args.all_models:
        models_to_eval = ModelRegistry.list_models()
    elif args.model:
        models_to_eval = [args.model]
    else:
        logger.error('Must specify --model or --all-models')
        return
    
    logger.info(f'Models to evaluate: {models_to_eval}')
    logger.info(f'Dataset: {args.dataset}')
    
    # Load dataset
    logger.info('Loading dataset...')
    try:
        test_dataset = get_dataset(args.dataset, args.data_path, {}, split='test')
        logger.info(f'Test dataset: {len(test_dataset)} samples')
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return
    
    # Evaluate each model
    results = {}
    
    for model_name in models_to_eval:
        logger.info(f'\n{"="*50}')
        logger.info(f'Evaluating {model_name}')
        logger.info(f'{"="*50}')
        
        try:
            # Create model
            config = {'model_name': model_name, 'device': args.device}
            model = ModelRegistry.get_model(model_name, config)
            model = model.to_device(device)
            
            # Load checkpoint
            if args.checkpoint:
                checkpoint_path = args.checkpoint
            else:
                checkpoint_path = args.checkpoint_dir / model_name / f'checkpoint-{args.dataset}.pt'
            
            if checkpoint_path.exists():
                logger.info(f'Loading checkpoint: {checkpoint_path}')
                model.load_checkpoint(checkpoint_path)
            elif args.pretrained:
                logger.warning(f'Pretrained checkpoint not found: {checkpoint_path}')
                logger.info('Skipping this model...')
                continue
            else:
                logger.warning('No checkpoint specified, using randomly initialized model')
            
            # Evaluate
            logger.info('Running evaluation...')
            start_time = time.time()
            
            metrics, predictions = evaluate_model(
                model, test_loader, device,
                detection_level=args.detection_level,
                k_neighbors=args.k_neighbors
            )
            
            eval_time = time.time() - start_time
            
            # Log results
            logger.info(f'\nResults for {model_name}:')
            logger.info('-' * 50)
            for metric_name, value in metrics.items():
                logger.info(f'{metric_name}: {value:.4f}')
            logger.info(f'Evaluation time: {format_time(eval_time)}')
            
            # Store results
            results[model_name] = {
                'metrics': metrics,
                'eval_time': eval_time,
                'checkpoint': str(checkpoint_path)
            }
            
            # Save predictions if requested
            if args.save_predictions and predictions:
                pred_file = args.output_dir / f'{model_name}_{args.dataset}_predictions.npy'
                np.save(pred_file, np.array(predictions))
                logger.info(f'Predictions saved: {pred_file}')
        
        except Exception as e:
            logger.error(f'Error evaluating {model_name}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall results
    results_file = args.output_dir / f'evaluation_results_{args.dataset}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved: {results_file}')
    
    # Print summary
    logger.info('\n' + '='*50)
    logger.info('Evaluation Summary')
    logger.info('='*50)
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Models evaluated: {len(results)}')
    
    if results:
        logger.info('\nModel Performance (AUC-ROC):')
        for model_name, result in sorted(results.items(), 
                                        key=lambda x: x[1]['metrics'].get('auc_roc', 0),
                                        reverse=True):
            auc = result['metrics'].get('auc_roc', 0)
            logger.info(f'  {model_name}: {auc:.4f}')
    
    logger.info('\nEvaluation finished!')


if __name__ == '__main__':
    main()
