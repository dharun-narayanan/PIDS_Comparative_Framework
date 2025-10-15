"""
ThreaTrace Model Wrapper for PIDS Comparative Framework
"""

from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

from models.base_model import BasePIDSModel, ModelRegistry

# Import ThreaTrace standalone implementation
from models.implementations.threatrace import (
    ThreaTraceModel as ThreaTraceCore,
    SketchGenerator,
    setup_threatrace_model
)


@ModelRegistry.register('threatrace')
class ThreaTraceModel(BasePIDSModel):
    """
    ThreaTrace: Node-Level Threat Detection via Provenance Graph Learning
    Paper: IEEE TIFS 2022
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get architecture configuration
        arch_config = config.get('architecture', {})
        
        # Model parameters
        in_channels = arch_config.get('in_channels', 64)
        hidden_channels = arch_config.get('hidden_channels', 128)
        out_channels = arch_config.get('out_channels', 64)
        encoder_type = arch_config.get('encoder_type', 'sage')
        num_layers = arch_config.get('num_layers', 2)
        dropout = arch_config.get('dropout', 0.5)
        sketch_size = arch_config.get('sketch_size', 256)
        num_classes = arch_config.get('num_classes', 2)
        
        # Build model using standalone implementation
        self.threatrace_model = ThreaTraceCore(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            encoder_type=encoder_type,
            num_layers=num_layers,
            dropout=dropout,
            sketch_size=sketch_size,
            num_classes=num_classes
        )
        
        # Move to device (device is already set in parent __init__)
        try:
            self.threatrace_model = self.threatrace_model.to(self.device)
        except RuntimeError as e:
            self.logger.warning(f"Could not move model to {self.device}: {e}. Using CPU.")
            self.device = torch.device('cpu')
            self.threatrace_model = self.threatrace_model.to(self.device)
        
        self.logger.info(f"ThreaTrace model initialized with {encoder_type} encoder on {self.device}")
    
    def forward(self, batch):
        """Forward pass through ThreaTrace model."""
        x = batch.x if hasattr(batch, 'x') else torch.randn(batch.num_nodes, 64).to(self.device)
        edge_index = batch.edge_index
        batch_assignment = getattr(batch, 'batch', None)
        
        return self.threatrace_model(x, edge_index, batch_assignment)
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train ThreaTrace for one epoch."""
        self.train()
        
        total_loss = 0
        num_batches = 0
        criterion = nn.CrossEntropyLoss()
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = self.forward(batch)
            
            # Compute loss
            if hasattr(batch, 'y'):
                loss = criterion(out, batch.y)
            else:
                # Unsupervised loss
                loss = torch.mean(out ** 2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate ThreaTrace model."""
        self.eval()
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.forward(batch)
                
                # Get predictions
                scores = torch.softmax(out, dim=-1)[:, 1]
                preds = torch.argmax(out, dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
                
                if hasattr(batch, 'y'):
                    all_labels.append(batch.y.cpu().numpy())
        
        # Compute metrics
        metrics = {}
        if len(all_labels) > 0:
            from utils.metrics import compute_detection_metrics
            
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_scores = np.concatenate(all_scores)
            
            metrics = compute_detection_metrics(all_labels, all_preds, all_scores)
        else:
            metrics = {'auc_roc': 0.0}
        
        return metrics
    
    def get_embeddings(self, batch):
        """Extract embeddings from ThreaTrace model."""
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            
            if hasattr(self, 'conv1'):
                x = batch.x if hasattr(batch, 'x') else torch.randn(batch.num_nodes, 128).to(self.device)
                edge_index = batch.edge_index
                
                x = torch.relu(self.conv1(x, edge_index))
                x = torch.relu(self.conv2(x, edge_index))
                return x
            else:
                x = batch.x if hasattr(batch, 'x') else torch.randn(batch.num_nodes, 128).to(self.device)
                return self.encoder(x)
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save ThreaTrace checkpoint."""
        checkpoint = {
            'config': self.config,
            **kwargs
        }
        
        if hasattr(self, 'conv1'):
            checkpoint['conv1_state_dict'] = self.conv1.state_dict()
            checkpoint['conv2_state_dict'] = self.conv2.state_dict()
            checkpoint['classifier_state_dict'] = self.classifier.state_dict()
        else:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
            checkpoint['classifier_state_dict'] = self.classifier.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load ThreaTrace checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'conv1_state_dict' in checkpoint:
            self.conv1.load_state_dict(checkpoint['conv1_state_dict'])
            self.conv2.load_state_dict(checkpoint['conv2_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    @staticmethod
    def load_pretrained(checkpoint_path: Path, config: Dict[str, Any]) -> 'ThreaTraceModel':
        """Load pretrained ThreaTrace model."""
        model = ThreaTraceModel(config)
        model.load_checkpoint(checkpoint_path)
        return model
