# PIDS Comparative Framework - Extension Guide

**Complete guide to adding new state-of-the-art PIDS models to the framework**

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Framework Architecture](#-framework-architecture)
3. [Quick Start: Adding a New Model](#-quick-start-adding-a-new-model)
4. [Step-by-Step Tutorial](#-step-by-step-tutorial)
5. [Model Implementation Requirements](#-model-implementation-requirements)
6. [Testing Your Model](#-testing-your-model)
7. [Best Practices](#-best-practices)
8. [Complete Integration Example](#-complete-integration-example)
9. [Troubleshooting](#-troubleshooting)
10. [FAQ](#-frequently-asked-questions)

---

## 🎯 Overview

The PIDS Comparative Framework is designed to **easily integrate new state-of-the-art PIDS models**. The plugin-based architecture allows you to add a new model with minimal code changes.

### What You Need

1. ✅ **Model implementation** (PyTorch-based)
2. ✅ **Wrapper class** (adapts your model to framework interface)
3. ✅ **Registration** (one-line decorator)
4. ✅ **Optional: Configuration file** (YAML)

**Time required:** 1-3 hours for a basic integration

### What You Get

- ✅ Automatic integration with evaluation pipeline
- ✅ Consistent metrics and comparison
- ✅ Support for pretrained weights
- ✅ Configuration management
- ✅ Logging and error handling
- ✅ GPU/CPU compatibility

---

## 🏗️ Framework Architecture

### Plugin System

The framework uses a **decorator-based plugin architecture** for model registration:

```python
@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    """Your model implementation."""
    pass
```

Models are automatically discovered and integrated when the framework loads.

### Component Structure

```
models/
├── base_model.py               # BasePIDSModel interface & ModelRegistry
├── __init__.py                 # Auto-discovery system
│
├── implementations/            # Standalone implementations
│   └── your_model/            # ← Your model implementation
│       ├── __init__.py        # Export main classes
│       ├── model.py           # Core model architecture
│       ├── layers.py          # Custom layers (optional)
│       └── utils.py           # Helper functions (optional)
│
└── your_model_wrapper.py      # ← Wrapper adapting to BasePIDSModel
```

### Optional Components

```
configs/
└── models/
    └── your_model.yaml         # Model configuration

checkpoints/
└── your_model/                 # Pretrained weights
    └── checkpoint-*.pt

experiments/
└── train_your_model.py         # Custom training script (optional)
```

---

## 🚀 Quick Start: Adding a New Model

### Step 1: Create Implementation Directory

```bash
cd PIDS_Comparative_Framework
mkdir -p models/implementations/your_model
cd models/implementations/your_model
```

### Step 2: Create Standalone Implementation

```python
# models/implementations/your_model/__init__.py
"""
YourModel Implementation
Standalone implementation for PIDS Comparative Framework.

Paper: "Your Paper Title" (Conference Year)
Authors: Your Name et al.
"""

from .model import YourModelCore
from .utils import your_helper_function

__all__ = ['YourModelCore', 'your_helper_function']
```

```python
# models/implementations/your_model/model.py
import torch
import torch.nn as nn

class YourModelCore(nn.Module):
    """Core implementation of YourModel for provenance-based IDS."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Example: Simple GNN encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Example: Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index=None):
        """Forward pass."""
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h  # Return reconstruction and embeddings
```

### Step 3: Create Wrapper Class

```python
# models/your_model_wrapper.py
import torch
import torch.nn.functional as F
from typing import Dict, Any
from pathlib import Path

from models.base_model import BasePIDSModel, ModelRegistry
from models.implementations.your_model import YourModelCore

@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    """YourModel wrapper for PIDS Comparative Framework."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract config parameters
        in_dim = config.get('in_dim', 128)
        hidden_dim = config.get('hidden_dim', 256)
        out_dim = config.get('out_dim', 128)
        num_layers = config.get('num_layers', 3)
        
        # Build model
        self.model = YourModelCore(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers
        ).to(self.device)
        
        self.logger.info(f"YourModel initialized with {self.count_parameters()} parameters")
    
    def forward(self, batch):
        """Forward pass through model."""
        x = batch['x'].to(self.device)
        edge_index = batch.get('edge_index', None)
        
        if edge_index is not None:
            edge_index = edge_index.to(self.device)
        
        reconstruction, embeddings = self.model(x, edge_index)
        return reconstruction, embeddings
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train for one epoch."""
        self.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, _ = self.forward(batch)
            
            # Compute loss (example: MSE reconstruction loss)
            target = batch['x'].to(self.device)
            loss = F.mse_loss(reconstruction, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate model."""
        self.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                reconstruction, embeddings = self.forward(batch)
                
                # Compute anomaly scores (example: reconstruction error)
                target = batch['x'].to(self.device)
                scores = F.mse_loss(reconstruction, target, reduction='none').mean(dim=1)
                
                all_scores.append(scores.cpu())
                if 'label' in batch:
                    all_labels.append(batch['label'].cpu())
        
        # Concatenate results
        scores = torch.cat(all_scores)
        
        if all_labels:
            labels = torch.cat(all_labels)
            
            # Compute metrics
            from utils.metrics import compute_metrics
            metrics = compute_metrics(labels, scores)
            return metrics
        else:
            return {'scores': scores.numpy()}
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded checkpoint from {path}")
```

### Step 4: Register in `models/__init__.py`

```python
# models/__init__.py
# Add to imports:
try:
    from models.your_model_wrapper import YourModel
except ImportError as e:
    print(f"Warning: Could not import YourModel: {e}")

# Add to __all__:
__all__ = [
    'BasePIDSModel',
    'ModelRegistry',
    'MAGICModel',
    'KairosModel',
    'OrthrusModel',
    'ThreaTraceModel',
    'ContinuumFLModel',
    'YourModel',  # ← Add this
]
```

### Step 5: Test Integration

```bash
# Test if model is registered
python -c "from models import ModelRegistry; print(ModelRegistry.list_models())"
# Expected output includes: 'your_model'

# Test model creation
python -c "from models import ModelRegistry; model = ModelRegistry.get_model('your_model', {}); print(model)"
```

### Step 6: Create Configuration (Optional)

```yaml
# configs/models/your_model.yaml
model:
  name: your_model
  type: your_architecture_type
  
  architecture:
    in_dim: 128
    hidden_dim: 256
    out_dim: 128
    num_layers: 3
    dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: adam
  scheduler: cosine
  weight_decay: 0.0001

evaluation:
  batch_size: 64
  k_neighbors: 5
  detection_level: entity  # or 'batch' or 'both'
  threshold: 0.5
```

### Step 7: Run Evaluation

```bash
# Evaluate your model
python experiments/evaluate.py \
    --model your_model \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --config configs/models/your_model.yaml
```

**Done! Your model is now integrated into the framework. 🎉**

---

## 📚 Step-by-Step Tutorial

### Understanding BasePIDSModel Interface

All models must implement the `BasePIDSModel` abstract class:

```python
class BasePIDSModel(ABC, nn.Module):
    """Base class for all PIDS models."""
    
    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Forward pass through model.
        
        Args:
            batch: Batch data (dict or DGLGraph or PyG Data)
        
        Returns:
            Model output (scores, embeddings, etc.)
        """
        pass
    
    @abstractmethod
    def train_epoch(self, dataloader, optimizer, **kwargs) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of metrics (e.g., {'loss': 0.5})
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataloader, **kwargs) -> Dict[str, float]:
        """Evaluate model.
        
        Args:
            dataloader: Evaluation data loader
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of metrics (e.g., {'auc_roc': 0.95, 'f1': 0.88})
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: Path, **kwargs) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path, **kwargs) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            **kwargs: Additional loading options
        """
        pass
    
    # Optional methods:
    def get_embeddings(self, batch) -> torch.Tensor:
        """Extract embeddings for entity-level detection.
        
        Args:
            batch: Batch data
        
        Returns:
            Embeddings tensor
        """
        raise NotImplementedError("get_embeddings not implemented")
```

### Creating Standalone Implementation

Your implementation should be **self-contained** without external repository dependencies.

```python
# models/implementations/your_model/model.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

class YourModelCore(nn.Module):
    """
    Core implementation of YourModel.
    
    Architecture:
        - Encoder: Graph Neural Network or Autoencoder
        - Decoder: Anomaly detection head
        - Loss: Reconstruction loss or contrastive loss
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output embedding dimension
        num_layers: Number of GNN/encoder layers
        dropout: Dropout rate
        activation: Activation function ('relu', 'elu', 'tanh')
    
    Example:
        >>> model = YourModelCore(in_dim=128, hidden_dim=256, out_dim=128)
        >>> x = torch.randn(100, 128)  # 100 nodes, 128 features
        >>> edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
        >>> out, embeddings = model(x, edge_index)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        # Build encoder layers
        encoder_layers = []
        current_dim = in_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:
                encoder_layers.append(self._get_activation(activation))
                encoder_layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (for reconstruction-based models)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def encode(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input to embeddings.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges] (optional)
        
        Returns:
            Embeddings [num_nodes, out_dim]
        """
        # If your model uses graph structure:
        if edge_index is not None:
            # Apply graph convolution or message passing
            # For simplicity, we use MLP here
            pass
        
        embeddings = self.encoder(x)
        return embeddings
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to reconstruction.
        
        Args:
            embeddings: Embeddings [num_nodes, out_dim]
        
        Returns:
            Reconstruction [num_nodes, in_dim]
        """
        reconstruction = self.decoder(embeddings)
        return reconstruction
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges] (optional)
        
        Returns:
            Tuple of (reconstruction, embeddings)
        """
        embeddings = self.encode(x, edge_index)
        reconstruction = self.decode(embeddings)
        return reconstruction, embeddings
```

### Implementing the Wrapper

The wrapper adapts your model to the framework interface:

```python
# models/your_model_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from pathlib import Path

from models.base_model import BasePIDSModel, ModelRegistry
from models.implementations.your_model import YourModelCore

@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    """YourModel wrapper for PIDS Comparative Framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)
        
        # Extract configuration
        architecture = config.get('architecture', {})
        in_dim = architecture.get('in_dim', 128)
        hidden_dim = architecture.get('hidden_dim', 256)
        out_dim = architecture.get('out_dim', 128)
        num_layers = architecture.get('num_layers', 3)
        dropout = architecture.get('dropout', 0.1)
        activation = architecture.get('activation', 'relu')
        
        # Build model
        self.model = YourModelCore(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        ).to(self.device)
        
        # Store hyperparameters
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.logger.info(f"YourModel initialized:")
        self.logger.info(f"  - Parameters: {self.count_parameters():,}")
        self.logger.info(f"  - Architecture: in_dim={in_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}")
        self.logger.info(f"  - Layers: {num_layers}, Dropout: {dropout}")
        self.logger.info(f"  - Device: {self.device}")
    
    def forward(self, batch):
        """Forward pass through model.
        
        Args:
            batch: Batch data (dict, DGLGraph, or PyG Data)
        
        Returns:
            Tuple of (reconstruction, embeddings)
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            x = batch['x'].to(self.device)
            edge_index = batch.get('edge_index', None)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
        else:
            # Handle DGL or PyG graphs
            x = batch.ndata['feat'].to(self.device) if hasattr(batch, 'ndata') else batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device) if hasattr(batch, 'edge_index') else None
        
        # Forward pass
        reconstruction, embeddings = self.model(x, edge_index)
        
        return reconstruction, embeddings
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            **kwargs: Additional arguments (e.g., scheduler, clip_grad)
        
        Returns:
            Dictionary of training metrics
        """
        self.train()
        
        total_loss = 0
        num_batches = 0
        
        # Optional gradient clipping
        clip_grad = kwargs.get('clip_grad', None)
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, embeddings = self.forward(batch)
            
            # Get target
            if isinstance(batch, dict):
                target = batch['x'].to(self.device)
            else:
                target = batch.ndata['feat'].to(self.device) if hasattr(batch, 'ndata') else batch.x.to(self.device)
            
            # Compute loss (example: MSE reconstruction loss)
            loss = F.mse_loss(reconstruction, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % 100 == 0:
                self.logger.debug(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate model.
        
        Args:
            dataloader: Evaluation data loader
            **kwargs: Additional arguments (e.g., k_neighbors, threshold)
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        
        all_scores = []
        all_labels = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                reconstruction, embeddings = self.forward(batch)
                
                # Get target and labels
                if isinstance(batch, dict):
                    target = batch['x'].to(self.device)
                    labels = batch.get('label', None)
                else:
                    target = batch.ndata['feat'].to(self.device) if hasattr(batch, 'ndata') else batch.x.to(self.device)
                    labels = batch.ndata.get('label', None) if hasattr(batch, 'ndata') else getattr(batch, 'y', None)
                
                # Compute anomaly scores (reconstruction error)
                scores = F.mse_loss(reconstruction, target, reduction='none').mean(dim=1)
                
                all_scores.append(scores.cpu())
                all_embeddings.append(embeddings.cpu())
                
                if labels is not None:
                    all_labels.append(labels.cpu())
        
        # Concatenate results
        scores = torch.cat(all_scores).numpy()
        embeddings = torch.cat(all_embeddings).numpy()
        
        if all_labels:
            labels = torch.cat(all_labels).numpy()
            
            # Compute metrics
            from utils.metrics import compute_metrics
            metrics = compute_metrics(labels, scores)
            
            self.logger.info(f"Evaluation Results:")
            self.logger.info(f"  - AUROC: {metrics['auc_roc']:.4f}")
            self.logger.info(f"  - AUPRC: {metrics['auc_prc']:.4f}")
            self.logger.info(f"  - F1-Score: {metrics['f1']:.4f}")
            
            return metrics
        else:
            return {'scores': scores, 'embeddings': embeddings}
    
    def get_embeddings(self, batch):
        """Extract embeddings for entity-level detection.
        
        Args:
            batch: Batch data
        
        Returns:
            Embeddings tensor [num_nodes, out_dim]
        """
        self.eval()
        with torch.no_grad():
            _, embeddings = self.forward(batch)
        return embeddings
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save (epoch, optimizer state, etc.)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'in_dim': self.in_dim,
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            **kwargs
        }
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            **kwargs: Additional loading options (strict, etc.)
        """
        strict = kwargs.get('strict', True)
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        self.logger.info(f"Loaded checkpoint from {path}")
        if 'epoch' in checkpoint:
            self.logger.info(f"  - Checkpoint epoch: {checkpoint['epoch']}")
```

---

## ✅ Model Implementation Requirements

### Mandatory Methods

1. **`__init__(config)`**: Initialize model with configuration
2. **`forward(batch)`**: Forward pass through model
3. **`train_epoch(dataloader, optimizer)`**: Train for one epoch
4. **`evaluate(dataloader)`**: Evaluate model and return metrics
5. **`save_checkpoint(path)`**: Save model state
6. **`load_checkpoint(path)`**: Load model state

### Optional Methods

- **`get_embeddings(batch)`**: Extract embeddings for entity-level detection
- **`predict(batch)`**: Make predictions (if different from evaluate)
- **`visualize(batch)`**: Visualize model internals (for debugging)

### Configuration Requirements

Your model should accept a configuration dictionary:

```python
config = {
    'architecture': {
        'in_dim': 128,
        'hidden_dim': 256,
        'out_dim': 128,
        'num_layers': 3,
        'dropout': 0.1
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100
    },
    'evaluation': {
        'batch_size': 64,
        'k_neighbors': 5
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

---

## 🧪 Testing Your Model

### Unit Tests

Create unit tests for your model:

```python
# tests/test_your_model.py
import unittest
import torch
from models import ModelRegistry

class TestYourModel(unittest.TestCase):
    def setUp(self):
        """Set up test model."""
        self.config = {
            'architecture': {
                'in_dim': 128,
                'hidden_dim': 256,
                'out_dim': 128
            }
        }
        self.model = ModelRegistry.get_model('your_model', self.config)
    
    def test_model_creation(self):
        """Test model can be created."""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'forward'))
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch = {
            'x': torch.randn(100, 128),  # 100 nodes, 128 features
            'edge_index': torch.randint(0, 100, (2, 500))  # 500 edges
        }
        
        reconstruction, embeddings = self.model(batch)
        
        self.assertEqual(reconstruction.shape, (100, 128))
        self.assertEqual(embeddings.shape, (100, 128))
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            
            # Save checkpoint
            self.model.save_checkpoint(checkpoint_path, epoch=1)
            self.assertTrue(checkpoint_path.exists())
            
            # Load checkpoint
            new_model = ModelRegistry.get_model('your_model', self.config)
            new_model.load_checkpoint(checkpoint_path)

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m unittest tests/test_your_model.py
```

### Integration Test

Test full evaluation workflow:

```bash
# Create dummy data
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/test_data \
    --dataset-name test

# Evaluate your model
python experiments/evaluate.py \
    --model your_model \
    --dataset test \
    --data-path data/test_data \
    --pretrained
```

---

## 🎓 Best Practices

### 1. Logging

Use framework logger for consistent logging:

```python
# Good
self.logger.info("Training epoch 1...")
self.logger.debug(f"Batch shape: {batch.shape}")
self.logger.warning("Missing labels, using unsupervised mode")
self.logger.error("CUDA out of memory")

# Avoid
print("Training epoch 1...")
```

### 2. Device Management

Always use `self.device` for device placement:

```python
# Good
batch = batch.to(self.device)
model = model.to(self.device)

# Avoid hardcoding
batch = batch.to('cuda')
```

### 3. Error Handling

Handle edge cases gracefully:

```python
def forward(self, batch):
    try:
        x = batch['x'].to(self.device)
    except KeyError:
        raise ValueError("Batch must contain 'x' key with node features")
    
    if x.dim() != 2:
        raise ValueError(f"Expected 2D features, got shape {x.shape}")
    
    return self.model(x)
```

### 4. Configuration Validation

Validate configuration early:

```python
def __init__(self, config):
    super().__init__(config)
    
    # Validate required config
    if 'architecture' not in config:
        raise ValueError("Config must contain 'architecture' section")
    
    arch = config['architecture']
    required_keys = ['in_dim', 'hidden_dim', 'out_dim']
    for key in required_keys:
        if key not in arch:
            raise ValueError(f"Architecture config must contain '{key}'")
```

### 5. Memory Management

For large datasets, manage memory carefully:

```python
def evaluate(self, dataloader, **kwargs):
    self.eval()
    
    # Don't accumulate all batches in memory
    # Process in chunks
    all_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            scores = self.forward(batch)
            all_scores.append(scores.cpu())  # Move to CPU immediately
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return torch.cat(all_scores)
```

---

## 💡 Complete Integration Example

See the **MAGIC model** as a complete reference implementation:

```
models/
├── implementations/
│   └── magic/
│       ├── __init__.py          # Exports: build_model, GMAEModel, GAT, etc.
│       ├── gat.py               # Graph Attention Network layers
│       ├── autoencoder.py       # Masked autoencoder implementation
│       ├── loss_func.py         # SCE loss and other losses
│       ├── eval.py              # Evaluation utilities (k-NN, metrics)
│       └── utils.py             # Helper functions
│
└── magic_wrapper.py             # Wrapper adapting to BasePIDSModel
```

**Study these files to understand the complete integration pattern:**

1. **Standalone implementation** (`implementations/magic/`)
2. **Wrapper class** (`magic_wrapper.py`)
3. **Registration** (`@ModelRegistry.register('magic')`)
4. **Configuration** (`configs/models/magic.yaml`)

---

## 🐛 Troubleshooting

### Issue 1: Model Not Found in Registry

**Problem:**
```
ValueError: Model 'your_model' not found in registry
```

**Solutions:**
1. Check decorator: Ensure `@ModelRegistry.register('your_model')` is present
2. Check imports: Verify `models/__init__.py` imports your wrapper
3. Restart Python: `import` caches modules, restart to reload

```python
# Verify registration
from models import ModelRegistry
print(ModelRegistry.list_models())  # Should include 'your_model'
```

---

### Issue 2: Forward Pass Errors

**Problem:**
```
TypeError: forward() got an unexpected keyword argument 'edge_index'
```

**Solution:** Use `**kwargs` for flexibility:

```python
def forward(self, batch, **kwargs):
    # Extract what you need
    x = batch.get('x', batch)
    edge_index = kwargs.get('edge_index', None)
    return self.model(x, edge_index)
```

---

### Issue 3: Checkpoint Loading Fails

**Problem:**
```
RuntimeError: Error loading state_dict
```

**Solutions:**
1. Use `strict=False` for partial loading:
```python
self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

2. Check architecture matches:
```python
# Save architecture info in checkpoint
checkpoint = {
    'model_state_dict': self.model.state_dict(),
    'architecture': {'in_dim': 128, 'hidden_dim': 256, ...}
}
```

---

### Issue 4: Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
```python
config['training']['batch_size'] = 8  # Reduce from 32
```

2. Use gradient accumulation:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = self.compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. Clear cache regularly:
```python
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## ❓ Frequently Asked Questions

### Q1: Can I use a non-PyTorch model?

**A:** The framework is PyTorch-centric, but you can wrap other frameworks:

```python
class ScikitLearnWrapper(BasePIDSModel):
    def __init__(self, config):
        super().__init__(config)
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(**config.get('model_params', {}))
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        # Collect all data
        X = []
        for batch in dataloader:
            X.append(batch['x'].cpu().numpy())
        X = np.vstack(X)
        
        # Train sklearn model
        self.model.fit(X)
        return {'loss': 0.0}  # No loss for sklearn
```

---

### Q2: How do I handle custom data formats?

**A:** Implement data format conversion in your forward method:

```python
def forward(self, batch):
    # Handle dict format
    if isinstance(batch, dict):
        x = batch['x']
        edge_index = batch.get('edge_index', None)
    
    # Handle DGL graph
    elif hasattr(batch, 'ndata'):
        x = batch.ndata['feat']
        edge_index = batch.edges()
    
    # Handle PyG graph
    elif hasattr(batch, 'x'):
        x = batch.x
        edge_index = batch.edge_index
    
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")
    
    return self.model(x, edge_index)
```

---

### Q3: Can I add custom metrics?

**A:** Yes, add them to your `evaluate` method:

```python
def evaluate(self, dataloader, **kwargs):
    # ... compute scores and labels ...
    
    from utils.metrics import compute_metrics
    metrics = compute_metrics(labels, scores)
    
    # Add custom metrics
    from sklearn.metrics import matthews_corrcoef
    metrics['mcc'] = matthews_corrcoef(labels, (scores > 0.5).astype(int))
    
    # Add your own
    metrics['custom_score'] = self.compute_custom_metric(labels, scores)
    
    return metrics
```

---

### Q4: How do I debug my model?

**A:** Use verbose logging and visualization:

```python
def forward(self, batch):
    x = batch['x'].to(self.device)
    
    # Debug logging
    self.logger.debug(f"Input shape: {x.shape}")
    self.logger.debug(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Check for NaNs
    if torch.isnan(x).any():
        self.logger.error("NaN detected in input!")
        raise ValueError("Input contains NaN values")
    
    embeddings = self.model.encode(x)
    
    # Debug embeddings
    self.logger.debug(f"Embedding shape: {embeddings.shape}")
    self.logger.debug(f"Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    return embeddings
```

---

### Q5: How do I use pretrained weights from external repos?

**A:** Load and convert weights in your wrapper:

```python
def load_external_checkpoint(self, path: Path):
    """Load checkpoint from external repository format."""
    # Load external checkpoint
    external_ckpt = torch.load(path, map_location=self.device)
    
    # Convert to your format
    state_dict = {}
    for key, value in external_ckpt['model'].items():
        # Rename keys if needed
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        state_dict[new_key] = value
    
    # Load converted state dict
    self.model.load_state_dict(state_dict, strict=False)
    self.logger.info(f"Loaded external checkpoint from {path}")
```

---

## 📋 Integration Checklist

Use this checklist to ensure complete integration:

- [ ] **Implementation created** in `models/implementations/your_model/`
- [ ] **Wrapper class created** inheriting from `BasePIDSModel`
- [ ] **Model registered** with `@ModelRegistry.register('your_model')`
- [ ] **All mandatory methods implemented** (forward, train_epoch, evaluate, save/load checkpoint)
- [ ] **Configuration file created** in `configs/models/your_model.yaml` (optional)
- [ ] **Unit tests written** in `tests/test_your_model.py`
- [ ] **Integration test passed** with `evaluate.py`
- [ ] **Documentation added** (docstrings in all methods)
- [ ] **Pretrained weights prepared** in `checkpoints/your_model/` (optional)
- [ ] **README updated** to list your model
- [ ] **Model registered in** `models/__init__.py`

---

## 🎯 Next Steps

After integrating your model:

1. **Test thoroughly** with dummy and real data
2. **Benchmark performance** against other models
3. **Document** model-specific usage and hyperparameters
4. **Share** your integration with the community
5. **Contribute** improvements back to the framework

---

## 📚 Additional Resources

- **Base Model API**: See `models/base_model.py`
- **Existing Examples**: `models/magic_wrapper.py`, `models/kairos_wrapper.py`, `models/orthrus_wrapper.py`
- **Dataset Handling**: See `data/dataset.py`
- **Metrics**: See `utils/metrics.py`
- **Framework Documentation**: See `README.md` and `Setup.md`

---

<div align="center">

**Congratulations! You've successfully integrated a new PIDS model! 🎉**

Your model is now available for evaluation alongside other state-of-the-art approaches.

[⬆ Back to Top](#pids-comparative-framework---extension-guide)

</div>
