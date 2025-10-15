"""
Kairos Model Wrapper for PIDS Comparative Framework
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from pathlib import Path

from models.base_model import BasePIDSModel, ModelRegistry
# Import from standalone implementation
from models.implementations.kairos import GraphAttentionEmbedding, LinkPredictor, TimeEncoder, setup_kairos_model, prepare_kairos_batch


@ModelRegistry.register('kairos')
class KairosModel(BasePIDSModel):
    """
    Kairos: Practical Intrusion Detection using Whole-system Provenance
    Paper: IEEE S&P 2024
    
    This wrapper uses a standalone implementation of Kairos that is self-contained
    within the framework and does not depend on external repositories.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.in_channels = config.get('in_channels', 100)
        self.out_channels = config.get('out_channels', 100)
        self.msg_dim = config.get('msg_dim', 100)
        
        # Initialize Kairos components using standalone implementation
        self.time_enc = TimeEncoder(self.out_channels)
        self.gnn = GraphAttentionEmbedding(
            self.in_channels,
            self.out_channels,
            self.msg_dim,
            self.time_enc
        )
        self.link_predictor = LinkPredictor(
            self.out_channels,
            self.out_channels
        )
        
        # Memory for temporal information
        self.memory_dim = config.get('memory_dim', 100)
        self.max_nodes = config.get('max_nodes', 268243)
        
        self.logger.info(f"Kairos model initialized with {self.count_parameters()} parameters")
    
    def forward(self, batch):
        """Forward pass through Kairos model."""
        # Extract batch components
        x = batch.x if hasattr(batch, 'x') else None
        edge_index = batch.edge_index if hasattr(batch, 'edge_index') else None
        t = batch.t if hasattr(batch, 't') else None
        msg = batch.msg if hasattr(batch, 'msg') else None
        last_update = batch.last_update if hasattr(batch, 'last_update') else t
        
        # Graph attention embedding
        h = self.gnn(x, last_update, edge_index, t, msg)
        
        # Link prediction
        pred = self.link_predictor(h[edge_index[0]], h[edge_index[1]])
        
        return pred
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train Kairos for one epoch."""
        self.train()
        
        total_loss = 0
        num_batches = 0
        criterion = nn.CrossEntropyLoss()
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = self.forward(batch)
            
            # Compute loss
            if hasattr(batch, 'y'):
                loss = criterion(pred, batch.y)
            else:
                # Unsupervised loss
                loss = self._compute_unsupervised_loss(pred, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def _compute_unsupervised_loss(self, pred, batch):
        """Compute unsupervised loss for Kairos."""
        # Placeholder - implement actual Kairos loss
        return torch.mean((pred - pred.mean()) ** 2)
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate Kairos model."""
        self.eval()
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                pred = self.forward(batch)
                
                # Get predictions and scores
                if pred.dim() > 1:
                    scores = torch.softmax(pred, dim=-1)[:, 1]
                    pred_labels = torch.argmax(pred, dim=-1)
                else:
                    scores = torch.sigmoid(pred)
                    pred_labels = (scores > 0.5).long()
                
                all_preds.append(pred_labels.cpu().numpy())
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
            metrics = {'predictions_made': len(all_preds)}
        
        return metrics
    
    def get_embeddings(self, batch):
        """Extract embeddings from Kairos model."""
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            x = batch.x if hasattr(batch, 'x') else None
            edge_index = batch.edge_index
            t = batch.t if hasattr(batch, 't') else None
            msg = batch.msg if hasattr(batch, 'msg') else None
            last_update = batch.last_update if hasattr(batch, 'last_update') else t
            
            h = self.gnn(x, last_update, edge_index, t, msg)
            return h
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save Kairos checkpoint."""
        checkpoint = {
            'gnn_state_dict': self.gnn.state_dict(),
            'link_predictor_state_dict': self.link_predictor.state_dict(),
            'time_enc_state_dict': self.time_enc.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load Kairos checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])
        self.time_enc.load_state_dict(checkpoint['time_enc_state_dict'])
        self.logger.info(f"Checkpoint loaded from {path}")
    
    @staticmethod
    def load_pretrained(checkpoint_path: Path, config: Dict[str, Any]) -> 'KairosModel':
        """Load pretrained Kairos model."""
        model = KairosModel(config)
        model.load_checkpoint(checkpoint_path)
        return model
