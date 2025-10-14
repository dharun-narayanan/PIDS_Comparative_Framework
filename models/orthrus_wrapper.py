"""
Orthrus Model Wrapper for PIDS Comparative Framework
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

# Add Orthrus directory to path
ORTHRUS_DIR = Path(__file__).parent.parent.parent / 'orthrus/src'
sys.path.insert(0, str(ORTHRUS_DIR))

from models.base_model import BasePIDSModel, ModelRegistry


@ModelRegistry.register('orthrus')
class OrthrusModel(BasePIDSModel):
    """
    Orthrus: High Quality Attribution in Provenance-based IDS
    Paper: USENIX Security 2025
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            from model import Orthrus
            from encoders import get_encoder
            from decoders import get_decoders
            
            # Model configuration
            self.num_nodes = config.get('num_nodes', 100000)
            self.in_dim = config.get('in_dim', 100)
            self.out_dim = config.get('out_dim', 100)
            self.use_contrastive = config.get('use_contrastive_learning', True)
            
            # Build encoder and decoders
            encoder = get_encoder(config)
            decoders = get_decoders(config)
            
            # Initialize Orthrus model
            self.orthrus_model = Orthrus(
                encoder=encoder,
                decoders=decoders,
                num_nodes=self.num_nodes,
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                use_contrastive_learning=self.use_contrastive,
                device=self.device,
                graph_reindexer=None
            )
            
            self.logger.info(f"Orthrus model initialized with {self.count_parameters()} parameters")
            
        except ImportError as e:
            self.logger.error(f"Failed to import Orthrus components: {e}")
            raise
    
    def forward(self, batch):
        """Forward pass through Orthrus model."""
        full_data = batch if hasattr(batch, 'full_data') else None
        return self.orthrus_model(batch, full_data, inference=False)
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train Orthrus for one epoch."""
        self.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = self.forward(batch)
            
            # Handle multi-task loss
            if isinstance(loss, dict):
                total_batch_loss = sum(loss.values())
            else:
                total_batch_loss = loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate Orthrus model."""
        self.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass in inference mode
                output = self.orthrus_model(batch, None, inference=True)
                
                # Extract scores
                if isinstance(output, dict):
                    scores = output.get('scores', output.get('predictions'))
                else:
                    scores = output
                
                if scores is not None:
                    all_scores.append(scores.cpu().numpy())
                
                if hasattr(batch, 'y'):
                    all_labels.append(batch.y.cpu().numpy())
        
        # Compute metrics
        metrics = {}
        if len(all_labels) > 0 and len(all_scores) > 0:
            from utils.metrics import compute_detection_metrics
            
            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            
            # Convert scores to predictions
            threshold = np.median(all_scores)
            all_preds = (all_scores > threshold).astype(int)
            
            metrics = compute_detection_metrics(all_labels, all_preds, all_scores)
        else:
            metrics = {'auc_roc': 0.0}
        
        return metrics
    
    def get_embeddings(self, batch):
        """Extract embeddings from Orthrus encoder."""
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            h = self.orthrus_model.encoder(
                edge_index=batch.edge_index,
                t=batch.t,
                x=(batch.x_src, batch.x_dst),
                msg=batch.msg if hasattr(batch, 'msg') else None,
                edge_feats=batch.edge_feats if hasattr(batch, 'edge_feats') else None,
                full_data=None,
                inference=True,
                edge_types=batch.edge_type if hasattr(batch, 'edge_type') else None
            )
            return h
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save Orthrus checkpoint."""
        checkpoint = {
            'model_state_dict': self.orthrus_model.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load Orthrus checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.orthrus_model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded from {path}")
    
    @staticmethod
    def load_pretrained(checkpoint_path: Path, config: Dict[str, Any]) -> 'OrthrusModel':
        """Load pretrained Orthrus model."""
        model = OrthrusModel(config)
        model.load_checkpoint(checkpoint_path)
        return model
