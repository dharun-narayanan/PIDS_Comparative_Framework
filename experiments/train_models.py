"""
PIDS Comparative Framework - Model Training Script

This script provides comprehensive training functionality for all PIDS models
on both DARPA TC and custom SOC datasets, supporting both supervised and
unsupervised learning approaches.

Features:
- Unified training interface for all models (MAGIC, Kairos, Continuum_FL, ThreatTrace, Orthrus)
- Support for supervised and unsupervised learning
- DARPA TC dataset training (CADETS, THEIA, TRACE, ClearScope)
- Custom SOC dataset training
- Automatic checkpoint saving and resuming
- Early stopping and learning rate scheduling
- Comprehensive logging and metrics tracking
- Multi-GPU support

Usage:
    # Train MAGIC on CADETS E3 (unsupervised)
    python experiments/train_models.py --model magic --dataset cadets_e3 --mode unsupervised

    # Train Kairos on custom SOC (supervised with labels)
    python experiments/train_models.py --model kairos --dataset custom_soc --mode supervised --labels-file data/custom_soc/labels.json
    
    # Resume training from checkpoint
    python experiments/train_models.py --model magic --dataset cadets_e3 --resume --checkpoint train_checkpoints/magic/checkpoint_epoch_10.pt
    
    # Train with custom configuration
    python experiments/train_models.py --model magic --dataset cadets_e3 --config configs/training/magic_custom.yaml
"""

import argparse
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_builder import ModelBuilder
from utils.graph_dataset import PreprocessedGraphDataset
from utils.common import (
    set_seed, setup_logging, load_config, save_config, save_json,
    get_device, ensure_dir, format_time, EarlyStopping, AverageMeter
)
from utils.metrics import compute_detection_metrics, MetricsTracker


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PIDS models on DARPA TC or custom SOC datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model and dataset arguments
    model_group = parser.add_argument_group('Model and Dataset')
    model_group.add_argument(
        '--model', type=str, required=True,
        choices=['magic', 'kairos', 'continuum_fl', 'threatrace', 'orthrus'],
        help='Model to train (REQUIRED)'
    )
    model_group.add_argument(
        '--dataset', type=str, required=False,
        help='Dataset name (cadets_e3, theia_e3, trace_e3, clearscope_e3, custom_soc). If using --config, this is optional.'
    )
    model_group.add_argument(
        '--data-path', type=Path, default=None,
        help='Path to dataset (auto-detected if not specified)'
    )
    model_group.add_argument(
        '--config', type=Path, default=None,
        help='Path to dataset config file (e.g., configs/training/cadets_e3.yaml). If provided, dataset name is taken from config.'
    )
    
    # Training mode (auto-detected, these are advanced options)
    mode_group = parser.add_argument_group('Training Mode (Advanced - Auto-detected by default)')
    mode_group.add_argument(
        '--mode', type=str, default='auto',
        choices=['auto', 'supervised', 'unsupervised', 'semi_supervised'],
        help='Training mode (default: auto - detects based on label availability)'
    )
    mode_group.add_argument(
        '--labels-file', type=Path, default=None,
        help='Path to labels file for supervised learning (JSON format). If not specified, checks config and dataset default locations.'
    )
    mode_group.add_argument(
        '--label-ratio', type=float, default=0.1,
        help='Ratio of labeled data for semi-supervised learning (default: 0.1)'
    )
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=1,
                            help='Batch size (1 for graph-level training)')
    train_group.add_argument('--lr', type=float, default=0.0005,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=0.0,
                            help='Weight decay (L2 regularization)')
    train_group.add_argument('--optimizer', type=str, default='adam',
                            choices=['adam', 'adamw', 'sgd'],
                            help='Optimizer')
    train_group.add_argument('--scheduler', type=str, default=None,
                            choices=['step', 'cosine', 'plateau'],
                            help='Learning rate scheduler')
    train_group.add_argument('--warmup-epochs', type=int, default=0,
                            help='Number of warmup epochs')
    
    # Model architecture (overrides config)
    arch_group = parser.add_argument_group('Model Architecture')
    arch_group.add_argument('--hidden-dim', type=int, default=None,
                           help='Hidden dimension size')
    arch_group.add_argument('--num-layers', type=int, default=None,
                           help='Number of layers')
    arch_group.add_argument('--dropout', type=float, default=None,
                           help='Dropout rate')
    arch_group.add_argument('--num-heads', type=int, default=None,
                           help='Number of attention heads (for GAT)')
    
    # Checkpoint and resuming
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument('--resume', action='store_true',
                                 help='Resume training from checkpoint')
    checkpoint_group.add_argument('--checkpoint', type=Path, default=None,
                                 help='Path to checkpoint file to resume from')
    checkpoint_group.add_argument('--pretrained', action='store_true',
                                 help='Start from pretrained weights (for fine-tuning)')
    checkpoint_group.add_argument('--save-freq', type=int, default=5,
                                 help='Save checkpoint every N epochs')
    checkpoint_group.add_argument('--save-dir', type=Path, default=Path('train_checkpoints'),
                                 help='Directory to save training checkpoints')
    
    # Training options
    options_group = parser.add_argument_group('Training Options')
    options_group.add_argument('--early-stopping', type=int, default=10,
                              help='Early stopping patience (0 to disable)')
    options_group.add_argument('--val-split', type=float, default=0.2,
                              help='Validation split ratio')
    options_group.add_argument('--grad-clip', type=float, default=1.0,
                              help='Gradient clipping value (0 to disable)')
    options_group.add_argument('--accumulation-steps', type=int, default=1,
                              help='Gradient accumulation steps')
    
    # Data preprocessing
    data_group = parser.add_argument_group('Data Preprocessing')
    data_group.add_argument('--force-preprocess', action='store_true',
                           help='Force preprocessing even if preprocessed data exists')
    data_group.add_argument('--preprocess-only', action='store_true',
                           help='Only preprocess data without training')
    data_group.add_argument('--max-events-per-file', type=int, default=None,
                           help='Maximum events to load per file (for testing/sampling)')
    data_group.add_argument('--max-files', type=int, default=5,
                           help='Maximum number of files to process')
    data_group.add_argument('--time-window', type=int, default=3600,
                           help='Time window size in seconds (for temporal data)')
    data_group.add_argument('--max-nodes', type=int, default=None,
                           help='Maximum nodes per graph (for memory efficiency)')
    data_group.add_argument('--num-workers', type=int, default=4,
                           help='Number of data loading workers')
    
    # System arguments
    system_group = parser.add_argument_group('System')
    system_group.add_argument('--device', type=int, default=-1,
                             help='Device: -1 for CPU, 0+ for GPU device ID')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    system_group.add_argument('--fp16', action='store_true',
                             help='Use mixed precision training (FP16)')
    
    # Logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log-dir', type=Path, default=Path('logs/training'),
                          help='Directory for logs')
    log_group.add_argument('--experiment-name', type=str, default=None,
                          help='Experiment name for logging')
    log_group.add_argument('--wandb', action='store_true',
                          help='Use Weights & Biases logging')
    log_group.add_argument('--debug', action='store_true',
                          help='Debug mode with verbose logging')
    
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build complete training configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Complete training configuration dictionary
    """
    # Load base config from file if provided
    if args.config:
        config = load_config(args.config)
        # Override dataset from config if not provided as argument
        if not args.dataset and 'dataset_name' in config:
            args.dataset = config['dataset_name']
    else:
        config = {}
    
    # Validate that we have a dataset
    if not args.dataset:
        raise ValueError("Dataset must be specified either via --dataset or in --config file")
    
    # Build config from arguments (overrides file config)
    config.update({
        # Model and dataset
        'model_name': args.model,
        'dataset_name': args.dataset,
        'data_path': str(args.data_path) if args.data_path else config.get('data_path'),
        
        # Training mode (will be auto-detected if 'auto')
        'mode': args.mode,
        'labels_file': str(args.labels_file) if args.labels_file else config.get('labels_file'),
        'label_ratio': args.label_ratio,
        
        # Training hyperparameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup_epochs,
        
        # Model architecture overrides
        'architecture_overrides': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'num_heads': args.num_heads,
        },
        
        # Training options
        'early_stopping_patience': args.early_stopping,
        'val_split': args.val_split,
        'grad_clip': args.grad_clip,
        'accumulation_steps': args.accumulation_steps,
        
        # Data preprocessing
        'force_preprocess': args.force_preprocess,
        'preprocess_only': args.preprocess_only,
        'max_events_per_file': args.max_events_per_file,
        'max_files': args.max_files,
        'time_window': args.time_window,
        'max_nodes': args.max_nodes,
        'num_workers': args.num_workers,
        'remove_self_loops': config.get('remove_self_loops', True),
        
        # Checkpointing
        'resume': args.resume,
        'checkpoint_path': str(args.checkpoint) if args.checkpoint else None,
        'pretrained': args.pretrained,
        'save_freq': args.save_freq,
        'save_dir': str(args.save_dir),
        
        # System
        'device': args.device,
        'seed': args.seed,
        'fp16': args.fp16,
        
        # Logging
        'log_dir': str(args.log_dir),
        'experiment_name': args.experiment_name,
        'wandb': args.wandb,
        'debug': args.debug,
    })
    
    # Remove None values from architecture overrides
    config['architecture_overrides'] = {
        k: v for k, v in config['architecture_overrides'].items() if v is not None
    }
    
    return config


def auto_detect_data_path(dataset_name: str) -> Optional[Path]:
    """
    Auto-detect data path based on dataset name.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to dataset or None if not found
    """
    # Check common locations
    base_dirs = [
        Path('.'),
        Path('..'),
        Path('../..'),
    ]
    
    # DARPA datasets
    if any(x in dataset_name for x in ['cadets', 'theia', 'trace', 'clearscope']):
        for base in base_dirs:
            darpa_dir = base / 'DARPA'
            if darpa_dir.exists():
                # Look for specific dataset folder
                for subdir in darpa_dir.iterdir():
                    if subdir.is_dir() and dataset_name.replace('_', '-') in subdir.name.lower():
                        return subdir
                return darpa_dir
    
    # Custom SOC dataset
    elif 'custom' in dataset_name.lower():
        for base in base_dirs:
            custom_dir = base / 'custom_dataset'
            if custom_dir.exists():
                return custom_dir
            
            # Also check in data/custom_soc
            custom_soc = base / 'data' / 'custom_soc'
            if custom_soc.exists():
                return custom_soc
    
    return None


def run_preprocessing(config: Dict[str, Any]) -> Path:
    """
    Run preprocessing to create graph file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to preprocessed graph file
    """
    from scripts.preprocess_data import UniversalPreprocessor
    
    logger.info("\n" + "="*80)
    logger.info("  Running Data Preprocessing")
    logger.info("="*80 + "\n")
    
    # Determine output path
    dataset_type = 'darpa' if any(x in config['dataset_name'] for x in ['cadets', 'theia', 'trace', 'clearscope']) else 'custom_soc'
    output_dir = Path('data') / dataset_type
    output_file = output_dir / f"{config['dataset_name']}_graph.pkl"
    
    # Check if preprocessing is needed
    if output_file.exists() and not config.get('force_preprocess', False):
        logger.info(f"✓ Preprocessed data already exists: {output_file}")
        logger.info("  Use --force-preprocess to reprocess")
        return output_file
    
    logger.info(f"Preprocessing {config['dataset_name']} dataset...")
    logger.info(f"Input: {config['data_path']}")
    logger.info(f"Output: {output_file}")
    
    # Build preprocessing config
    preprocess_config = {
        'dataset_type': dataset_type,
        'data': {
            'time_window': config.get('time_window', 3600),
            'max_events_per_file': config.get('max_events_per_file'),
            'max_files': config.get('max_files', 5),
        },
        'graph': {
            'remove_self_loops': config.get('remove_self_loops', True),
        },
        'preprocessing': {
            'remove_self_loops': config.get('remove_self_loops', True),
        }
    }
    
    # Create preprocessor
    preprocessor = UniversalPreprocessor(preprocess_config)
    
    # Find data files
    data_path = Path(config['data_path'])
    if data_path.is_file():
        input_files = [data_path]
    else:
        # Look for JSON, NDJSON, and binary AVRO files
        input_files = []
        for pattern in ['*.json', '*.ndjson', '*.bin', '*.avro']:
            input_files.extend(list(data_path.glob(pattern)))
            # Also check subdirectories
            for subdir in data_path.glob('*'):
                if subdir.is_dir():
                    input_files.extend(list(subdir.glob(pattern)))
        input_files = sorted(set(input_files))
    
    if not input_files:
        raise FileNotFoundError(f"No data files found in {data_path}")
    
    logger.info(f"Found {len(input_files)} data files")
    
    # Load and parse files
    events = preprocessor.load_and_parse_files(
        input_files,
        max_events_per_file=config.get('max_events_per_file')
    )
    
    if not events:
        raise ValueError("No events were parsed from input files")
    
    # Build graphs (with or without time windows)
    graphs = preprocessor.build_graphs_with_time_windows(events)
    
    if not graphs:
        raise ValueError("No graphs were created during preprocessing")
    
    # Save graphs
    preprocessor.save_graphs(graphs, output_file)
    
    # Print statistics
    preprocessor.print_statistics()
    
    logger.info(f"\n✓ Preprocessing complete: {output_file}\n")
    
    return output_file


def create_dataloaders(
    dataset: PreprocessedGraphDataset,
    config: Dict[str, Any]
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        dataset: Training dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split into train and validation
    val_split = config.get('val_split', 0.2)
    
    # If dataset is too small for splitting, use all for training
    if len(dataset) < 5 or val_split <= 0:
        train_dataset = dataset
        val_dataset = None
        if len(dataset) < 5:
            logger.warning(f"Dataset too small ({len(dataset)} samples) for train/val split. Using all samples for training.")
        else:
            logger.info(f"No validation split, using all {len(dataset)} samples for training")
    else:
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        # Ensure at least 1 sample in each split
        if train_size == 0:
            train_size = 1
            val_size = len(dataset) - 1
        elif val_size == 0:
            val_size = 1
            train_size = len(dataset) - 1
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config['seed'])
        )
        
        logger.info(f"Split dataset: {train_size} train, {val_size} validation")
    
    # Create dataloaders
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    optimizer_name = config['optimizer'].lower()
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Created optimizer: {optimizer_name}, LR={lr}, Weight Decay={weight_decay}")
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = config.get('scheduler')
    
    if scheduler_name is None:
        return None
    
    if scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    logger.info(f"Created scheduler: {scheduler_name}")
    return scheduler


def compute_loss(
    model: nn.Module,
    data: Any,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss based on training mode.
    
    Args:
        model: Model instance
        data: Batch data
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    mode = config['mode']
    metrics = {}
    
    # Move data to device
    if hasattr(data, 'to'):
        data = data.to(device)
    
    if mode == 'unsupervised':
        # Unsupervised training: reconstruction loss
        embeddings = model.encode(data)
        
        if model.multi_decoder:
            # Use reconstruction decoder
            output = model.decode(embeddings, data, decoder_name='reconstruction', inference=False)
        else:
            output = model.decode(embeddings, data, inference=False)
        
        # Compute reconstruction loss
        if isinstance(output, dict) and 'loss' in output:
            loss = output['loss']
            metrics.update(output.get('metrics', {}))
        elif isinstance(output, torch.Tensor) and output.ndim == 0:
            # Scalar loss
            loss = output
        else:
            # Compute MSE reconstruction loss
            reconstructed = output
            original = data.x
            loss = nn.functional.mse_loss(reconstructed, original)
        
        metrics['reconstruction_loss'] = loss.item()
    
    elif mode == 'supervised':
        # Supervised training: classification loss
        embeddings = model.encode(data)
        
        if model.multi_decoder:
            # Use classification decoder
            output = model.decode(embeddings, data, decoder_name='classifier', inference=False)
        else:
            output = model.decode(embeddings, data, inference=False)
        
        # Compute classification loss
        if isinstance(output, dict) and 'loss' in output:
            loss = output['loss']
            metrics.update(output.get('metrics', {}))
        elif isinstance(output, torch.Tensor) and output.ndim == 0:
            loss = output
        else:
            # Compute cross-entropy loss
            labels = data.y
            if labels is not None:
                loss = nn.functional.cross_entropy(output, labels)
                
                # Compute accuracy
                pred = output.argmax(dim=1)
                acc = (pred == labels).float().mean()
                metrics['accuracy'] = acc.item()
            else:
                raise ValueError("Supervised mode requires labels in data.y")
        
        metrics['classification_loss'] = loss.item()
    
    elif mode == 'semi_supervised':
        # Semi-supervised: combine reconstruction and classification
        embeddings = model.encode(data)
        
        # Reconstruction loss (on all data)
        recon_output = model.decode(embeddings, data, decoder_name='reconstruction', inference=False)
        if isinstance(recon_output, dict):
            recon_loss = recon_output['loss']
        else:
            recon_loss = nn.functional.mse_loss(recon_output, data.x)
        
        # Classification loss (on labeled data only)
        if hasattr(data, 'labeled_mask') and data.labeled_mask.sum() > 0:
            class_output = model.decode(embeddings, data, decoder_name='classifier', inference=False)
            
            if isinstance(class_output, dict):
                class_loss = class_output['loss']
            else:
                labeled_output = class_output[data.labeled_mask]
                labeled_targets = data.y[data.labeled_mask]
                class_loss = nn.functional.cross_entropy(labeled_output, labeled_targets)
            
            # Combine losses
            loss = recon_loss + class_loss
            metrics['reconstruction_loss'] = recon_loss.item()
            metrics['classification_loss'] = class_loss.item()
        else:
            # No labeled data in this batch, use only reconstruction
            loss = recon_loss
            metrics['reconstruction_loss'] = recon_loss.item()
    
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    
    return loss, metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model instance
        train_loader: Training data loader
        optimizer: Optimizer instance
        config: Configuration dictionary
        device: Device to use
        epoch: Current epoch number
        scaler: GradScaler for mixed precision (optional)
        
    Returns:
        Dictionary of average metrics
    """
    model.train()
    
    # Metrics tracking
    loss_meter = AverageMeter()
    metrics_tracker = MetricsTracker()
    
    # Gradient accumulation
    accumulation_steps = config.get('accumulation_steps', 1)
    grad_clip = config.get('grad_clip', 0)
    
    # Progress bar (leave=False to update in place)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, data in enumerate(pbar):
        try:
            # Compute loss
            if config['fp16'] and scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    loss, batch_metrics = compute_loss(model, data, config, device)
                    loss = loss / accumulation_steps
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
            else:
                # Normal training
                loss, batch_metrics = compute_loss(model, data, config, device)
                loss = loss / accumulation_steps
                loss.backward()
            
            # Gradient accumulation and update
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Update metrics
            loss_meter.update(loss.item() * accumulation_steps)
            metrics_tracker.update(**batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                **{k: f"{v:.4f}" for k, v in list(batch_metrics.items())[:2]}
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            if config.get('debug'):
                import traceback
                traceback.print_exc()
            continue
    
    # Get average metrics
    avg_metrics = metrics_tracker.get_averages()
    avg_metrics['loss'] = loss_meter.avg
    
    return avg_metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model instance
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    
    # Metrics tracking
    loss_meter = AverageMeter()
    metrics_tracker = MetricsTracker()
    
    # Progress bar (leave=False to update in place)
    pbar = tqdm(val_loader, desc=f"Validation {epoch}", leave=False)
    
    for batch_idx, data in enumerate(pbar):
        try:
            # Compute loss
            loss, batch_metrics = compute_loss(model, data, config, device)
            
            # Update metrics
            loss_meter.update(loss.item())
            metrics_tracker.update(**batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                **{k: f"{v:.4f}" for k, v in list(batch_metrics.items())[:2]}
            })
            
        except Exception as e:
            logger.error(f"Error in validation batch {batch_idx}: {e}")
            if config.get('debug'):
                import traceback
                traceback.print_exc()
            continue
    
    # Get average metrics
    avg_metrics = metrics_tracker.get_averages()
    avg_metrics['loss'] = loss_meter.avg
    
    return avg_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_path: Path,
    is_best: bool = False
):
    """
    Save training checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance (optional)
        epoch: Current epoch
        metrics: Current metrics (dict with 'train' and 'val' keys)
        config: Configuration dictionary
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'is_best': is_best,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint (overwrites previous)
    ensure_dir(checkpoint_path.parent)
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        logger.info(f"Saved best checkpoint: {checkpoint_path}")
    else:
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[int, Dict[str, float]]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optimizer instance (optional)
        scheduler: Scheduler instance (optional)
        device: Device to load on
        
    Returns:
        Tuple of (start_epoch, metrics)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Resumed from epoch {start_epoch-1}")
    logger.info(f"Previous metrics: {metrics}")
    
    return start_epoch, metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Build configuration
    config = build_training_config(args)
    
    # Setup logging
    experiment_name = config['experiment_name'] or f"{config['model_name']}_{config['dataset_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config['log_dir']) / experiment_name
    ensure_dir(log_dir)
    
    log_level = logging.DEBUG if config['debug'] else logging.INFO
    logger = setup_logging(log_dir, level=log_level)
    
    logger.info('=' * 80)
    logger.info('PIDS Comparative Framework - Model Training')
    logger.info('=' * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Dataset: {config['dataset_name']}")
    logger.info(f"Mode: {config['mode']}")
    logger.info('=' * 80)
    
    # Set random seed
    set_seed(config['seed'])
    logger.info(f"Random seed: {config['seed']}")
    
    # Get device
    device = get_device(config['device'])
    logger.info(f"Device: {device}")
    
    # Save configuration
    config_path = log_dir / 'config.yaml'
    save_config(config, config_path)
    logger.info(f"Configuration saved: {config_path}")
    
    # Validate data path is provided
    if config['data_path'] is None:
        logger.error("Data path is required but not provided")
        logger.error("Please specify --data-path or --config with data_path")
        logger.error("Example: python experiments/train_models.py --model magic --dataset cadets_e3 --data-path ../DARPA/ta1-cadets-e3-official-1.json")
        return
    
    # Check if preprocessing is needed or force preprocess
    dataset_type = 'darpa' if any(x in config['dataset_name'] for x in ['cadets', 'theia', 'trace', 'clearscope']) else 'custom_soc'
    preprocessed_graph_path = Path('data') / dataset_type / f"{config['dataset_name']}_graph.pkl"
    
    # Run preprocessing if needed
    if not preprocessed_graph_path.exists() or config.get('force_preprocess', False):
        preprocessed_graph_path = run_preprocessing(config)
        
        # If --preprocess-only flag is set, exit after preprocessing
        if config.get('preprocess_only', False):
            logger.info("\n✓ Preprocessing complete (--preprocess-only flag set)")
            logger.info("Run training without --preprocess-only to train the model")
            return
    else:
        logger.info(f"Using existing preprocessed data: {preprocessed_graph_path}")
        logger.info("Use --force-preprocess to reprocess the data")
    
    # Load preprocessed dataset
    logger.info('\nLoading preprocessed dataset...')
    try:
        labels_path = config.get('labels_file')
        if labels_path:
            labels_path = Path(labels_path)
        
        dataset = PreprocessedGraphDataset(
            graph_path=preprocessed_graph_path,
            mode=config['mode'],
            labels_path=labels_path
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) == 0:
            logger.error("Dataset is empty! No graphs were loaded.")
            logger.error("Please check the preprocessed graph file or use --force-preprocess to regenerate")
            return
        
        logger.info(f"Dataset metadata: {dataset.get_metadata()}")
        
        # Auto-detect training mode based on label availability
        if config['mode'] == 'auto':
            if hasattr(dataset, 'labels') and len(dataset.labels) > 0 and any(label != 0 for label in dataset.labels):
                config['mode'] = 'supervised'
                logger.info("Auto-detected SUPERVISED training mode (labels found)")
            else:
                config['mode'] = 'unsupervised'
                logger.info("Auto-detected UNSUPERVISED training mode (no labels)")
        else:
            logger.info(f"Using specified training mode: {config['mode']}")
        
        # Update dataset mode
        dataset.mode = config['mode']
        
        # Get actual input dimension from dataset
        sample = dataset[0]
        if hasattr(sample, 'x'):
            actual_input_dim = sample.x.shape[-1]
            logger.info(f"Detected input dimension from data: {actual_input_dim}")
        else:
            actual_input_dim = None
            logger.warning("Could not detect input dimension from data")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        if config.get('debug'):
            import traceback
            traceback.print_exc()
        return
    
    # Create dataloaders
    logger.info('Creating dataloaders...')
    train_loader, val_loader = create_dataloaders(dataset, config)
    logger.info(f"Train batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info('Creating model...')
    try:
        model_builder = ModelBuilder(config_dir="configs/models")
        
        # Apply architecture overrides
        override_config = None
        if config['architecture_overrides']:
            override_config = {'architecture': config['architecture_overrides']}
        
        # Build model with input dimension adaptation
        model = model_builder.build_model(
            model_name=config['model_name'],
            override_config=override_config,
            input_dim=actual_input_dim
        )
        
        model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created: {config['model_name']}")
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {num_trainable:,}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        if config.get('debug'):
            import traceback
            traceback.print_exc()
        return
    
    # Load pretrained weights if requested
    if config['pretrained']:
        pretrained_path = Path(f"checkpoints/{config['model_name']}/checkpoint-{config['dataset_name']}.pt")
        if pretrained_path.exists():
            logger.info(f"Loading pretrained weights: {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location=device)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
                logger.info("Pretrained weights loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
        else:
            logger.warning(f"Pretrained weights not found: {pretrained_path}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create GradScaler for mixed precision training
    scaler = None
    if config['fp16']:
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled (FP16)")
        else:
            logger.warning("FP16 requested but CUDA not available, using FP32")
    
    # Setup early stopping
    early_stopping = None
    if config['early_stopping_patience'] > 0:
        early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            mode='min'
        )
        logger.info(f"Early stopping enabled (patience={config['early_stopping_patience']})")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config['resume'] and config['checkpoint_path']:
        checkpoint_path = Path(config['checkpoint_path'])
        if checkpoint_path.exists():
            start_epoch, prev_metrics = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            best_val_loss = prev_metrics.get('val_loss', float('inf'))
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
    
    # Training loop
    logger.info('Starting training...')
    logger.info('=' * 80)
    
    training_start_time = time.time()
    metrics_history = []
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        
        logger.info(f"\nEpoch {epoch + 1}/{config['epochs']}")
        logger.info('-' * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, device, epoch + 1, scaler
        )
        
        # Validate
        if val_loader is not None:
            val_metrics = validate_epoch(
                model, val_loader, config, device, epoch + 1
            )
        else:
            val_metrics = {}
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = val_metrics.get('loss', train_metrics['loss'])
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch + 1} completed in {format_time(epoch_time)}")
        logger.info(f"Learning rate: {current_lr:.6f}")
        logger.info(f"Train metrics: {train_metrics}")
        if val_metrics:
            logger.info(f"Validation metrics: {val_metrics}")
        
        # Track metrics history
        epoch_metrics = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'time': epoch_time
        }
        metrics_history.append(epoch_metrics)
        
        # Save checkpoint (only one checkpoint per dataset, overwrite each time)
        val_loss = val_metrics.get('loss', train_metrics['loss'])
        is_best = val_loss < best_val_loss
        
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        # Always save the latest checkpoint (overwrites previous)
        checkpoint_dir = Path(config['save_dir']) / config['model_name']
        checkpoint_path = checkpoint_dir / f"{config['dataset_name']}.pt"
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            {'train': train_metrics, 'val': val_metrics},
            config, checkpoint_path, is_best
        )
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Training completed
    total_training_time = time.time() - training_start_time
    logger.info('=' * 80)
    logger.info(f"Training completed in {format_time(total_training_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save metrics history
    metrics_path = log_dir / 'metrics_history.json'
    save_json(metrics_history, metrics_path)
    logger.info(f"Metrics history saved: {metrics_path}")
    
    # Generate training summary
    summary = {
        'experiment_name': experiment_name,
        'model': config['model_name'],
        'dataset': config['dataset_name'],
        'mode': config['mode'],
        'total_epochs': epoch + 1,
        'best_val_loss': best_val_loss,
        'total_time': total_training_time,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics,
        'config': config
    }
    
    summary_path = log_dir / 'training_summary.json'
    save_json(summary, summary_path)
    logger.info(f"Training summary saved: {summary_path}")
    
    logger.info('Training finished successfully!')


if __name__ == '__main__':
    main()
