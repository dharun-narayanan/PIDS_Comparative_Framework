"""
MAGIC Model Wrapper for PIDS Comparative Framework
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any

# Add MAGIC directory to path
MAGIC_DIR = Path(__file__).parent.parent.parent / 'MAGIC'
sys.path.insert(0, str(MAGIC_DIR))

from models.base_model import BasePIDSModel, ModelRegistry


@ModelRegistry.register('magic')
class MAGICModel(BasePIDSModel):
    """
    MAGIC: Masked Graph Representation Learning for APT Detection
    Paper: USENIX Security 2024
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Import MAGIC components
        from model.autoencoder import build_model as build_magic_model
        from utils.config import build_args
        
        # Build MAGIC model
        args = self._build_magic_args(config)
        self.magic_model = build_magic_model(args)
        self.args = args
        
        self.logger.info(f"MAGIC model initialized with {self.count_parameters()} parameters")
    
    def _build_magic_args(self, config):
        """Build MAGIC arguments from config."""
        class Args:
            pass
        
        args = Args()
        args.num_hidden = config.get('num_hidden', 256)
        args.num_layers = config.get('num_layers', 4)
        args.negative_slope = config.get('negative_slope', 0.2)
        args.mask_rate = config.get('mask_rate', 0.3)
        args.alpha_l = config.get('alpha_l', 2)
        args.n_dim = config.get('n_dim', 128)
        args.e_dim = config.get('e_dim', 64)
        args.device = config.get('device', 0)
        
        return args
    
    def forward(self, batch):
        """Forward pass through MAGIC model."""
        return self.magic_model(batch)
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train MAGIC for one epoch."""
        self.magic_model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            loss = self.magic_model(batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate MAGIC model."""
        self.magic_model.eval()
        
        from model.eval import evaluate_entity_level_using_knn
        import numpy as np
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                embeddings = self.magic_model.embed(batch)
                all_embeddings.append(embeddings.cpu().numpy())
                
                if hasattr(batch, 'y'):
                    all_labels.append(batch.y.cpu().numpy())
        
        if len(all_embeddings) > 0:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            
            if len(all_labels) > 0:
                all_labels = np.concatenate(all_labels, axis=0)
                # Compute metrics using k-NN
                metrics = {}
                # Placeholder - implement actual evaluation
                metrics['auc_roc'] = 0.0
            else:
                metrics = {'embeddings_computed': True}
        else:
            metrics = {'error': 'No data processed'}
        
        return metrics
    
    def get_embeddings(self, batch):
        """Extract embeddings from MAGIC model."""
        self.magic_model.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            return self.magic_model.embed(batch)
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save MAGIC checkpoint."""
        checkpoint = {
            'model_state_dict': self.magic_model.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load MAGIC checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.magic_model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded from {path}")
    
    @staticmethod
    def load_pretrained(checkpoint_path: Path, config: Dict[str, Any]) -> 'MAGICModel':
        """Load pretrained MAGIC model."""
        model = MAGICModel(config)
        model.load_checkpoint(checkpoint_path)
        return model


# Register model variants
@ModelRegistry.register('magic_streamspot')
class MAGICStreamSpot(MAGICModel):
    """MAGIC model optimized for StreamSpot dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        config.update({
            'num_hidden': 256,
            'num_layers': 4,
            'mask_rate': 0.3
        })
        super().__init__(config)


@ModelRegistry.register('magic_darpa')
class MAGICDARPA(MAGICModel):
    """MAGIC model optimized for DARPA TC datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        config.update({
            'num_hidden': 64,
            'num_layers': 3,
            'mask_rate': 0.5
        })
        super().__init__(config)
