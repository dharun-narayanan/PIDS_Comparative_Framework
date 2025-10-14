"""
Continuum_FL Model Wrapper

This module wraps the Continuum_FL (Federated Learning PIDS) model
into the framework's BasePIDSModel interface.

Paper: Continuum_FL - Federated Learning for Provenance-based IDS
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

# Add Continuum_FL to path
continuum_fl_path = Path(__file__).parent.parent.parent / "Continuum_FL"
sys.path.insert(0, str(continuum_fl_path))

from models.base_model import BasePIDSModel, ModelRegistry

# Import Continuum_FL components
try:
    from model.model import STGNN_AutoEncoder
    from utils.config import build_args
    from utils.poolers import Pooling
    from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
except ImportError as e:
    print(f"Warning: Could not import Continuum_FL modules: {e}")
    STGNN_AutoEncoder = None


@ModelRegistry.register('continuum_fl')
class ContinuumFLModel(BasePIDSModel):
    """
    Continuum_FL: Federated Learning PIDS Model
    
    Spatiotemporal Graph Neural Network with AutoEncoder architecture
    designed for federated learning scenarios.
    
    Architecture:
    - STGNN Encoder: Multi-layer GAT with temporal modeling
    - STGNN Decoder: Reconstruction decoder
    - RNN Cells: Temporal aggregation across snapshots
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Build Continuum_FL model
        self._build_model()
        
        # Setup pooling
        pooling_method = config.get('architecture', {}).get('pooling', 'mean')
        self.pooler = Pooling(pooling_method)
        
    def _build_model(self):
        """Build Continuum_FL STGNN AutoEncoder model."""
        arch_config = self.config.get('architecture', {})
        
        # Model dimensions
        n_dim = arch_config.get('n_dim', 100)
        e_dim = arch_config.get('e_dim', 64)
        hidden_dim = arch_config.get('hidden_dim', 256)
        out_dim = arch_config.get('out_dim', 64)
        
        # Architecture parameters
        n_layers = arch_config.get('n_layers', 4)
        n_heads = arch_config.get('n_heads', 4)
        n_snapshot = arch_config.get('n_snapshot', 10)
        
        # Regularization
        feat_drop = arch_config.get('feat_drop', 0.1)
        negative_slope = arch_config.get('negative_slope', 0.2)
        
        # Other parameters
        activation = arch_config.get('activation', 'prelu')
        residual = arch_config.get('residual', True)
        norm = arch_config.get('norm', 'BatchNorm')
        pooling = arch_config.get('pooling', 'mean')
        loss_fn = arch_config.get('loss_fn', 'sce')
        alpha_l = arch_config.get('alpha_l', 2)
        use_all_hidden = arch_config.get('use_all_hidden', True)
        
        # Build model
        if STGNN_AutoEncoder is not None:
            self.continuum_model = STGNN_AutoEncoder(
                n_dim=n_dim,
                e_dim=e_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                device=self.device,
                number_snapshot=n_snapshot,
                activation=activation,
                feat_drop=feat_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                pooling=pooling,
                loss_fn=loss_fn,
                alpha_l=alpha_l,
                use_all_hidden=use_all_hidden
            ).to(self.device)
        else:
            # Fallback model if Continuum_FL not available
            print("Warning: Using fallback model (Continuum_FL not available)")
            self.continuum_model = self._build_fallback_model(n_dim, out_dim)
    
    def _build_fallback_model(self, in_dim: int, out_dim: int):
        """Build a simple fallback model if Continuum_FL is not available."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, in_dim)
        ).to(self.device)
    
    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch of graph snapshots (list of DGL graphs)
            
        Returns:
            Loss tensor for the batch
        """
        # Move batch to device
        if isinstance(batch, list):
            graphs = [g.to(self.device) for g in batch]
        else:
            graphs = batch.to(self.device)
        
        # Forward pass
        if hasattr(self.continuum_model, 'forward'):
            loss = self.continuum_model(graphs)
            return loss
        else:
            # Fallback for simple model
            # Assume batch has features
            features = batch[0].ndata['attr'].float().to(self.device)
            reconstructed = self.continuum_model(features)
            loss = nn.MSELoss()(reconstructed, features)
            return loss
    
    def train_epoch(self, dataloader, optimizer) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Dictionary with training metrics
        """
        self.continuum_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            loss = self.forward(batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.continuum_model.eval()
        
        # Extract embeddings for all graphs
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Get embeddings
                embeddings = self.get_embeddings(batch)
                
                # Get labels (if available)
                if isinstance(batch, list) and len(batch) > 0:
                    if hasattr(batch[0], 'graph_label'):
                        labels = torch.tensor([g.graph_label for g in batch])
                        all_labels.append(labels)
                
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        if len(all_embeddings) > 0:
            embeddings_tensor = torch.cat(all_embeddings, dim=0)
            
            # If we have labels, compute k-NN based anomaly detection
            if len(all_labels) > 0:
                labels_tensor = torch.cat(all_labels, dim=0)
                
                # Use k-NN for anomaly detection
                from utils.metrics import compute_detection_metrics
                
                # Compute distances (anomaly scores)
                distances = torch.cdist(embeddings_tensor, embeddings_tensor)
                k = min(20, distances.shape[0] - 1)
                knn_distances, _ = torch.topk(distances, k=k+1, largest=False, dim=1)
                anomaly_scores = knn_distances[:, 1:].mean(dim=1)  # Exclude self
                
                # Compute metrics
                metrics = compute_detection_metrics(
                    labels_tensor.cpu().numpy(),
                    anomaly_scores.cpu().numpy()
                )
                
                return metrics
        
        # Return default metrics if evaluation not possible
        return {
            'auroc': 0.0,
            'auprc': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    def get_embeddings(self, batch) -> torch.Tensor:
        """
        Get embeddings for a batch of graphs.
        
        Args:
            batch: Batch of graphs
            
        Returns:
            Embedding tensor
        """
        self.continuum_model.eval()
        
        with torch.no_grad():
            # Move batch to device
            if isinstance(batch, list):
                graphs = [g.to(self.device) for g in batch]
            else:
                graphs = batch.to(self.device)
            
            # Get embeddings
            if hasattr(self.continuum_model, 'embed'):
                embeddings = self.continuum_model.embed(graphs)
                
                # Pool embeddings if multiple snapshots
                if isinstance(embeddings, list):
                    # Average across snapshots
                    embeddings = torch.stack([e.mean(dim=0) for e in embeddings])
                
                return embeddings
            else:
                # Fallback: use encoder output
                node_features = []
                for g in (graphs if isinstance(graphs, list) else [graphs]):
                    node_features.append(g.ndata['attr'].float())
                
                # Simple mean pooling
                embeddings = torch.stack([f.mean(dim=0) for f in node_features])
                return embeddings
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.continuum_model.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.continuum_model.load_state_dict(checkpoint['model_state_dict'])


@ModelRegistry.register('continuum_fl_streamspot')
class ContinuumFLStreamSpot(ContinuumFLModel):
    """Continuum_FL variant optimized for StreamSpot dataset."""
    
    def __init__(self, config: Dict):
        # Override defaults for StreamSpot
        config['architecture'] = config.get('architecture', {})
        config['architecture'].setdefault('n_layers', 5)
        config['architecture'].setdefault('out_dim', 64)
        config['architecture'].setdefault('n_snapshot', 6)
        config['architecture'].setdefault('use_all_hidden', True)
        super().__init__(config)


@ModelRegistry.register('continuum_fl_darpa')
class ContinuumFLDARPA(ContinuumFLModel):
    """Continuum_FL variant optimized for DARPA datasets."""
    
    def __init__(self, config: Dict):
        # Override defaults for DARPA
        config['architecture'] = config.get('architecture', {})
        config['architecture'].setdefault('n_layers', 4)
        config['architecture'].setdefault('out_dim', 64)
        config['architecture'].setdefault('n_snapshot', 50)
        config['architecture'].setdefault('use_all_hidden', False)
        super().__init__(config)
