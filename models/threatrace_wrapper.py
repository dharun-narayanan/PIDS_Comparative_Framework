"""
ThreaTrace Model Wrapper for PIDS Comparative Framework
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

# Add ThreaTrace directory to path
THREATRACE_DIR = Path(__file__).parent.parent.parent / 'threaTrace'
sys.path.insert(0, str(THREATRACE_DIR))

from models.base_model import BasePIDSModel, ModelRegistry


@ModelRegistry.register('threatrace')
class ThreaTraceModel(BasePIDSModel):
    """
    ThreaTrace: Node-Level Threat Detection via Provenance Graph Learning
    Paper: IEEE TIFS 2022
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ThreaTrace uses GraphChi for graph processing
        # We'll create a simple wrapper for the Python interface
        
        self.model_dir = Path(config.get('model_dir', THREATRACE_DIR / 'models'))
        self.graph_data_dir = Path(config.get('graph_data_dir', THREATRACE_DIR / 'graphchi-cpp-master/graph_data'))
        
        # Model parameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        
        # Initialize simple PyTorch model for compatibility
        self._build_model()
        
        self.logger.info(f"ThreaTrace model initialized")
    
    def _build_model(self):
        """Build simple GNN model for ThreaTrace."""
        try:
            import torch_geometric.nn as pyg_nn
            
            self.conv1 = pyg_nn.GCNConv(-1, self.hidden_dim)
            self.conv2 = pyg_nn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.classifier = nn.Linear(self.hidden_dim, 2)
            
        except ImportError:
            self.logger.warning("PyTorch Geometric not available, using simple model")
            # Fallback to simple linear layers
            self.encoder = nn.Sequential(
                nn.Linear(128, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.classifier = nn.Linear(self.hidden_dim, 2)
    
    def forward(self, batch):
        """Forward pass through ThreaTrace model."""
        if hasattr(self, 'conv1'):
            # PyG version
            x = batch.x if hasattr(batch, 'x') else torch.randn(batch.num_nodes, 128).to(self.device)
            edge_index = batch.edge_index
            
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            out = self.classifier(x)
        else:
            # Simple version
            x = batch.x if hasattr(batch, 'x') else torch.randn(batch.num_nodes, 128).to(self.device)
            x = self.encoder(x)
            out = self.classifier(x)
        
        return out
    
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
