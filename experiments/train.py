"""
Training script for PIDS models.

Usage:
    python experiments/train.py --model magic --dataset custom --data-path data/custom
    python experiments/train.py --model kairos --dataset cadets-e3 --pretrained --fine-tune
"""

import argparse
import sys
from pathlib import Path
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from models import ModelRegistry
from data.dataset import get_dataset
from utils.common import (
    set_seed, setup_logging, load_config, save_config, 
    get_device, ensure_dir, format_time, EarlyStopping
)
from utils.metrics import MetricsTracker, compute_detection_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PIDS models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['magic', 'kairos', 'orthrus', 'threatrace', 'continuum_fl'],
                       help='Model to train')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (custom, cadets-e3, theia-e3, etc.)')
    parser.add_argument('--data-path', type=Path, default=Path('data/custom'),
                       help='Path to dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer')
    
    # Model configuration
    parser.add_argument('--config', type=Path, default=None,
                       help='Path to model config file')
    parser.add_argument('--num-hidden', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Pretrained model
    parser.add_argument('--pretrained', action='store_true',
                       help='Load pretrained weights')
    parser.add_argument('--checkpoint', type=Path, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Fine-tune pretrained model')
    
    # Training options
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (0 to disable)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--device', type=int, default=-1,
                       help='Device: -1 for CPU (default), 0+ for GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=Path, default=Path('logs'),
                       help='Directory for logs')
    parser.add_argument('--save-dir', type=Path, default=Path('checkpoints'),
                       help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    
    return parser.parse_args()


def build_config(args):
    """Build configuration from arguments."""
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        'model_name': args.model,
        'dataset_name': args.dataset,
        'data_path': str(args.data_path),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'num_hidden': args.num_hidden,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'device': args.device,
        'seed': args.seed,
    })
    
    return config


def train_epoch(model, dataloader, optimizer, device, logger):
    """Train for one epoch."""
    model.set_train_mode()
    
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device (implementation depends on data format)
        # batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - implementation depends on model
        try:
            loss = model(batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        except Exception as e:
            logger.error(f'Error in batch {batch_idx}: {e}')
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return {'loss': avg_loss}


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    ensure_dir(args.log_dir)
    logger = setup_logging(args.log_dir, 
                          level=logging.DEBUG if args.debug else logging.INFO)
    
    logger.info('='*50)
    logger.info('PIDS Model Training')
    logger.info('='*50)
    logger.info(f'Model: {args.model}')
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Data path: {args.data_path}')
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f'Random seed: {args.seed}')
    
    # Get device
    device = get_device(args.device)
    logger.info(f'Device: {device}')
    
    # Build configuration
    config = build_config(args)
    
    # Save configuration
    exp_dir = args.log_dir / f'{args.model}_{args.dataset}_{int(time.time())}'
    ensure_dir(exp_dir)
    save_config(config, exp_dir / 'config.yaml')
    
    # Load dataset
    logger.info('Loading dataset...')
    try:
        train_dataset = get_dataset(args.dataset, args.data_path, config, split='train')
        logger.info(f'Train dataset: {len(train_dataset)} samples')
        logger.info(f'Dataset metadata: {train_dataset.get_metadata()}')
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return
    
    # Create model
    logger.info('Creating model...')
    try:
        model = ModelRegistry.get_model(args.model, config)
        model = model.to_device(device)
        logger.info(f'Model created: {model.get_model_info()}')
    except Exception as e:
        logger.error(f'Failed to create model: {e}')
        logger.error(f'Make sure model {args.model} is properly registered')
        return
    
    # Load pretrained weights
    if args.pretrained or args.checkpoint:
        checkpoint_path = args.checkpoint or (args.save_dir / args.model / f'checkpoint-{args.dataset}.pt')
        if checkpoint_path.exists():
            logger.info(f'Loading pretrained weights from {checkpoint_path}')
            try:
                model.load_checkpoint(checkpoint_path)
            except Exception as e:
                logger.error(f'Failed to load checkpoint: {e}')
                return
        else:
            logger.warning(f'Checkpoint not found: {checkpoint_path}')
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    logger.info(f'Optimizer: {args.optimizer}, LR: {args.lr}')
    
    # Early stopping
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, mode='min')
        logger.info(f'Early stopping enabled with patience {args.early_stopping}')
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    logger.info('Starting training...')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        logger.info(f'\nEpoch {epoch+1}/{args.epochs}')
        logger.info('-' * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, logger)
        
        epoch_time = time.time() - epoch_start
        logger.info(f'Train Loss: {train_metrics["loss"]:.4f}, Time: {format_time(epoch_time)}')
        
        # Update metrics
        metrics_tracker.update(train_loss=train_metrics['loss'])
        
        # Early stopping check
        if early_stopping and early_stopping(train_metrics['loss']):
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = args.save_dir / args.model / f'checkpoint-{args.dataset}-epoch{epoch+1}.pt'
            ensure_dir(save_path.parent)
            model.save_checkpoint(save_path, epoch=epoch+1, optimizer=optimizer.state_dict())
            logger.info(f'Checkpoint saved: {save_path}')
    
    total_time = time.time() - start_time
    logger.info(f'\nTraining completed in {format_time(total_time)}')
    
    # Save final model
    final_path = args.save_dir / args.model / f'checkpoint-{args.dataset}.pt'
    ensure_dir(final_path.parent)
    model.save_checkpoint(final_path)
    logger.info(f'Final model saved: {final_path}')
    
    logger.info('Training finished!')


if __name__ == '__main__':
    main()
