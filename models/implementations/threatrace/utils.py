"""
Utility Functions for ThreaTrace
"""

import torch
import torch.nn as nn
import numpy as np
import random


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_threatrace_model(config):
    """
    Factory function to create ThreaTrace model
    
    Args:
        config: Model configuration dictionary with keys:
            - in_channels: Input feature dimension
            - hidden_channels: Hidden layer dimension
            - out_channels: Output embedding dimension
            - encoder_type: Encoder type ('sage' or 'gat')
            - num_layers: Number of GNN layers
            - dropout: Dropout rate
            - sketch_size: Sketch representation size
            - num_classes: Number of output classes
            
    Returns:
        model: ThreaTraceModel instance
    """
    from .model import ThreaTraceModel
    
    model = ThreaTraceModel(
        in_channels=config.get('in_channels', 64),
        hidden_channels=config.get('hidden_channels', 128),
        out_channels=config.get('out_channels', 64),
        encoder_type=config.get('encoder_type', 'sage'),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.5),
        sketch_size=config.get('sketch_size', 256),
        num_classes=config.get('num_classes', 2)
    )
    
    return model


def prepare_threatrace_batch(data):
    """
    Prepare batch data for ThreaTrace model
    
    Args:
        data: PyG Data or Batch object
        
    Returns:
        x: Node features
        edge_index: Edge indices
        batch: Batch assignment (or None for single graph)
    """
    x = data.x
    edge_index = data.edge_index
    batch = getattr(data, 'batch', None)
    
    return x, edge_index, batch


def compute_anomaly_score(model, x, edge_index, batch=None, threshold=0.5):
    """
    Compute anomaly scores for graphs
    
    Args:
        model: ThreaTrace model
        x: Node features
        edge_index: Edge indices
        batch: Batch assignment (optional)
        threshold: Classification threshold
        
    Returns:
        scores: Anomaly scores
        predictions: Binary predictions
    """
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, batch)
        probs = torch.softmax(logits, dim=1)
        
        # Anomaly score is probability of anomalous class
        scores = probs[:, 1] if logits.size(1) > 1 else probs[:, 0]
        predictions = (scores > threshold).long()
    
    return scores, predictions


def evaluate_threatrace(model, dataloader, device):
    """
    Evaluate ThreaTrace model
    
    Args:
        model: ThreaTrace model
        dataloader: Data loader
        device: Device to run on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            x, edge_index, batch = prepare_threatrace_batch(data)
            
            logits = model(x, edge_index, batch)
            
            if hasattr(data, 'y'):
                labels = data.y
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_correct': total_correct,
        'total_samples': total_samples
    }
    
    return metrics


def train_threatrace_epoch(model, dataloader, optimizer, device):
    """
    Train ThreaTrace model for one epoch
    
    Args:
        model: ThreaTrace model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        metrics: Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for data in dataloader:
        data = data.to(device)
        x, edge_index, batch = prepare_threatrace_batch(data)
        
        optimizer.zero_grad()
        logits = model(x, edge_index, batch)
        
        if hasattr(data, 'y'):
            labels = data.y
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy
    }
    
    return metrics
