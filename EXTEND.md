# PIDS Comparative Framework - Extension Guide

Complete guide to adding new state-of-the-art PIDS models to the framework.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Framework Extension Architecture](#framework-extension-architecture)
3. [Quick Start: Adding a New Model](#quick-start-adding-a-new-model)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Model Implementation Requirements](#model-implementation-requirements)
6. [Testing Your Model](#testing-your-model)
7. [Best Practices](#best-practices)
8. [Example: Complete Model Integration](#example-complete-model-integration)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The PIDS Comparative Framework is designed to **easily accept new state-of-the-art PIDS models**. Adding a new model requires:

1. ✅ Creating a standalone implementation
2. ✅ Writing a wrapper class
3. ✅ Registering with the framework
4. ✅ (Optional) Adding configuration

**Time required:** 1-3 hours for a basic integration

---

## 🏗️ Framework Extension Architecture

### Plugin System

The framework uses a **decorator-based plugin architecture**:

```python
@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    # Your implementation
    pass
```

Models are automatically discovered and integrated into the framework.

### Required Components

```
models/
├── implementations/
│   └── your_model/              # ← Your standalone implementation
│       ├── __init__.py          # Export main classes
│       ├── model.py             # Core model architecture
│       ├── layers.py            # Model layers (optional)
│       └── utils.py             # Helper functions (optional)
│
└── your_model_wrapper.py        # ← Wrapper adapting to BasePIDSModel
```

### Optional Components

```
configs/
└── models/
    └── your_model.yaml          # Model configuration

checkpoints/
└── your_model/                  # Pretrained weights
    └── checkpoint-dataset.pt

scripts/
└── train_your_model.py          # Custom training script (optional)
```

---

## 🚀 Quick Start: Adding a New Model

### 1. Create Implementation Directory

```bash
mkdir -p models/implementations/your_model
cd models/implementations/your_model
```

### 2. Create Standalone Implementation

```python
# models/implementations/your_model/__init__.py
"""
YourModel Implementation
Standalone implementation for PIDS Comparative Framework.

Paper: "Your Paper Title"
Conference/Journal Year
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
    """Core implementation of your PIDS model."""
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        h = torch.relu(self.encoder(x))
        out = self.decoder(h)
        return out
```

### 3. Create Wrapper Class

```python
# models/your_model_wrapper.py
import torch
from typing import Dict, Any
from pathlib import Path

from models.base_model import BasePIDSModel, ModelRegistry
from models.implementations.your_model import YourModelCore

@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    """YourModel wrapper for PIDS framework."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Build model from config
        self.model = YourModelCore(
            in_dim=config.get('in_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            out_dim=config.get('out_dim', 64)
        ).to(self.device)
        
        self.logger.info(f"YourModel initialized with {self.count_parameters()} parameters")
    
    def forward(self, batch):
        """Forward pass."""
        return self.model(batch)
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train for one epoch."""
        self.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            loss = self.forward(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return {'loss': total_loss / len(dataloader)}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate model."""
        self.eval()
        # Implement evaluation logic
        return {'auc_roc': 0.0}
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            **kwargs
        }, path)
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded checkpoint from {path}")
```

### 4. Register in `models/__init__.py`

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
    'YourModel',  # ← Add this
]
```

### 5. Test Integration

```bash
# Test if model is registered
python -c "from models import ModelRegistry; print(ModelRegistry.list_models())"
# Should output: ['magic', 'kairos', 'orthrus', 'threatrace', 'continuum_fl', 'your_model']

# Test model creation
python -c "from models import ModelRegistry; model = ModelRegistry.get_model('your_model', {}); print(model)"
```

### 6. Run Evaluation

```bash
python experiments/evaluate.py \
    --model your_model \
    --dataset custom \
    --data-path data/custom
```

**Done! Your model is now integrated into the framework.**

---

## 📚 Step-by-Step Tutorial

### Step 1: Understand BasePIDSModel Interface

All models must implement the `BasePIDSModel` interface:

```python
class BasePIDSModel(ABC, nn.Module):
    """Base class for all PIDS models."""
    
    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Forward pass through model."""
        pass
    
    @abstractmethod
    def train_epoch(self, dataloader, optimizer, **kwargs) -> Dict[str, float]:
        """Train for one epoch. Returns metrics dict."""
        pass
    
    @abstractmethod
    def evaluate(self, dataloader, **kwargs) -> Dict[str, float]:
        """Evaluate model. Returns metrics dict."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: Path, **kwargs) -> None:
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path, **kwargs) -> None:
        """Load model checkpoint."""
        pass
    
    # Optional methods:
    def get_embeddings(self, batch) -> torch.Tensor:
        """Extract embeddings (for entity-level detection)."""
        raise NotImplementedError
```

### Step 2: Create Standalone Implementation

Your implementation should be **self-contained** without external dependencies.

```python
# models/implementations/your_model/model.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class YourModelCore(nn.Module):
    """
    Core implementation of YourModel.
    
    Architecture:
        - Encoder: Graph Neural Network
        - Decoder: Anomaly detection head
        - Loss: Contrastive loss
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output embedding dimension
        num_layers: Number of GNN layers
    """
    
    def __init__(
        self, 
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Build encoder
        self.encoder = self._build_encoder(in_dim, hidden_dim, num_layers)
        
        # Build decoder
        self.decoder = self._build_decoder(hidden_dim, out_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _build_encoder(self, in_dim, hidden_dim, num_layers):
        """Build encoder layers."""
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*layers)
    
    def _build_decoder(self, hidden_dim, out_dim):
        """Build decoder layers."""
        return nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index=None, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges] (optional)
            
        Returns:
            out: Model output [num_nodes, out_dim]
        """
        # Encode
        h = self.encoder(x)
        
        # Decode
        out = self.decoder(h)
        
        return out
    
    def embed(self, x, edge_index=None):
        """Extract embeddings."""
        return self.encoder(x)
```

### Step 3: Create Wrapper Class

The wrapper adapts your implementation to the framework interface:

```python
# models/your_model_wrapper.py
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from pathlib import Path

from models.base_model import BasePIDSModel, ModelRegistry
from models.implementations.your_model import YourModelCore


@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    """
    YourModel: Brief Description
    Paper: "Your Paper Title" (Conference Year)
    
    This wrapper integrates YourModel into the PIDS Comparative Framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract architecture config
        arch_config = config.get('architecture', {})
        
        # Build model
        self.core_model = YourModelCore(
            in_dim=arch_config.get('in_dim', 128),
            hidden_dim=arch_config.get('hidden_dim', 256),
            out_dim=arch_config.get('out_dim', 64),
            num_layers=arch_config.get('num_layers', 3)
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        self.logger.info(f"YourModel initialized with {self.count_parameters()} parameters")
    
    def forward(self, batch):
        """Forward pass through the model."""
        # Extract batch components
        x = batch.x if hasattr(batch, 'x') else batch
        edge_index = getattr(batch, 'edge_index', None)
        
        # Forward pass
        out = self.core_model(x, edge_index)
        
        return out
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train for one epoch."""
        self.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            batch = batch.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = self.forward(batch)
            
            # Compute loss
            if hasattr(batch, 'y'):
                loss = self.criterion(out, batch.y.float())
            else:
                # Unsupervised loss
                loss = self._compute_unsupervised_loss(out)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate the model."""
        self.eval()
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.forward(batch)
                
                # Get scores and predictions
                scores = out.squeeze()
                preds = (scores > 0.5).long()
                
                all_scores.append(scores.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
                if hasattr(batch, 'y'):
                    all_labels.append(batch.y.cpu().numpy())
        
        # Compute metrics
        if len(all_labels) > 0:
            from utils.metrics import compute_detection_metrics
            
            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            
            metrics = compute_detection_metrics(all_labels, all_preds, all_scores)
        else:
            metrics = {'auc_roc': 0.0}
        
        return metrics
    
    def get_embeddings(self, batch):
        """Extract embeddings from the model."""
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            x = batch.x if hasattr(batch, 'x') else batch
            edge_index = getattr(batch, 'edge_index', None)
            embeddings = self.core_model.embed(x, edge_index)
            return embeddings
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.core_model.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.core_model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded from {path}")
    
    @staticmethod
    def load_pretrained(checkpoint_path: Path, config: Dict[str, Any]) -> 'YourModel':
        """Load a pretrained model."""
        model = YourModel(config)
        model.load_checkpoint(checkpoint_path)
        return model
    
    def _compute_unsupervised_loss(self, out):
        """Compute unsupervised loss (placeholder)."""
        # Implement your unsupervised loss here
        return torch.mean(out ** 2)
```

### Step 4: Add Configuration

Create a configuration file for your model:

```yaml
# configs/models/your_model.yaml

model_name: your_model

architecture:
  in_dim: 128
  hidden_dim: 256
  out_dim: 64
  num_layers: 3

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  optimizer: adam

evaluation:
  detection_level: entity
  k_neighbors: 5

# Pretrained checkpoint paths
checkpoints:
  cadets: checkpoints/your_model/checkpoint-cadets.pt
  streamspot: checkpoints/your_model/checkpoint-streamspot.pt
```

### Step 5: Add Helper Utilities (Optional)

```python
# models/implementations/your_model/utils.py

def setup_your_model(config):
    """Factory function to create YourModel from config."""
    from .model import YourModelCore
    return YourModelCore(
        in_dim=config.get('in_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        out_dim=config.get('out_dim', 64),
        num_layers=config.get('num_layers', 3)
    )

def preprocess_for_your_model(data):
    """Preprocess data for YourModel."""
    # Implement preprocessing logic
    return data

def postprocess_predictions(predictions):
    """Postprocess model predictions."""
    # Implement postprocessing logic
    return predictions
```

---

## ✅ Model Implementation Requirements

### Mandatory Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `forward(batch)` | Forward pass through model | Tensor (predictions/loss) |
| `train_epoch(dataloader, optimizer)` | Train for one epoch | Dict with metrics |
| `evaluate(dataloader)` | Evaluate on data | Dict with metrics |
| `save_checkpoint(path)` | Save model state | None |
| `load_checkpoint(path)` | Load model state | None |

### Optional Methods

| Method | Description | When to Implement |
|--------|-------------|-------------------|
| `get_embeddings(batch)` | Extract embeddings | For entity-level detection |
| `load_pretrained(path, config)` | Load pretrained model | For easier loading |

### Expected Metrics

Your `evaluate()` method should return a dictionary with:

```python
{
    'auc_roc': float,    # Area under ROC curve
    'auc_pr': float,     # Area under PR curve
    'f1': float,         # F1 score
    'precision': float,  # Precision
    'recall': float,     # Recall
    'fpr': float,        # False positive rate (optional)
}
```

---

## 🧪 Testing Your Model

### 1. Unit Tests

Create unit tests for your model:

```python
# tests/test_your_model.py
import pytest
import torch
from models import ModelRegistry

def test_model_registration():
    """Test that model is registered."""
    assert 'your_model' in ModelRegistry.list_models()

def test_model_creation():
    """Test model creation."""
    config = {'in_dim': 128, 'hidden_dim': 256}
    model = ModelRegistry.get_model('your_model', config)
    assert model is not None

def test_forward_pass():
    """Test forward pass."""
    config = {}
    model = ModelRegistry.get_model('your_model', config)
    
    # Create dummy input
    batch = torch.randn(32, 128)
    
    # Forward pass
    out = model(batch)
    
    assert out is not None
    assert out.shape[0] == 32

def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    from pathlib import Path
    
    config = {}
    model = ModelRegistry.get_model('your_model', config)
    
    # Save
    path = Path('test_checkpoint.pt')
    model.save_checkpoint(path)
    
    # Load
    model2 = ModelRegistry.get_model('your_model', config)
    model2.load_checkpoint(path)
    
    # Cleanup
    path.unlink()
```

Run tests:

```bash
pytest tests/test_your_model.py -v
```

### 2. Integration Tests

Test with the evaluation script:

```bash
# Test with dummy data
python experiments/evaluate.py \
    --model your_model \
    --dataset custom \
    --data-path data/custom \
    --debug
```

### 3. Benchmarking

Compare with existing models:

```bash
python experiments/compare.py \
    --models magic your_model \
    --dataset custom \
    --pretrained
```

---

## 📖 Best Practices

### 1. Code Organization

- ✅ Keep implementation self-contained in `models/implementations/your_model/`
- ✅ Separate core model from framework adapter
- ✅ Use clear, descriptive class/function names
- ✅ Add docstrings to all public methods

### 2. Configuration

- ✅ Use config dictionaries for all hyperparameters
- ✅ Provide sensible defaults
- ✅ Document all config options

### 3. Error Handling

```python
def forward(self, batch):
    try:
        # Forward pass logic
        out = self.model(batch)
        return out
    except Exception as e:
        self.logger.error(f"Error in forward pass: {e}")
        raise
```

### 4. Logging

```python
# Use framework logger
self.logger.info("Training epoch 1...")
self.logger.debug(f"Batch shape: {batch.shape}")
self.logger.warning("Missing labels, using unsupervised loss")
self.logger.error("CUDA out of memory")
```

### 5. Device Management

```python
# Always use self.device
batch = batch.to(self.device)
model = model.to(self.device)

# Handle both CPU and GPU
def __init__(self, config):
    super().__init__(config)
    self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 📝 Example: Complete Model Integration

See the `MAGIC` model as a reference example:

```
models/
├── implementations/
│   └── magic/
│       ├── __init__.py          # Exports main classes
│       ├── gat.py               # GAT implementation
│       ├── autoencoder.py       # Masked autoencoder
│       ├── loss_func.py         # Loss functions
│       ├── eval.py              # Evaluation utilities
│       └── utils.py             # Helper functions
│
└── magic_wrapper.py             # Wrapper class
```

**Study these files to understand the integration pattern.**

---

## 🐛 Troubleshooting

### Model Not Found

**Problem:**
```
ValueError: Model 'your_model' not found in registry
```

**Solution:**
- Check that `@ModelRegistry.register('your_model')` decorator is present
- Ensure `models/__init__.py` imports your wrapper
- Restart Python to reload modules

---

### Forward Pass Errors

**Problem:**
```
TypeError: forward() got an unexpected keyword argument 'edge_index'
```

**Solution:**
- Use `**kwargs` in forward signature: `def forward(self, batch, **kwargs)`
- Handle different batch formats gracefully

---

### Checkpoint Loading Fails

**Problem:**
```
RuntimeError: Error loading state_dict
```

**Solution:**
- Check that checkpoint structure matches model architecture
- Use `strict=False` in `load_state_dict()` for partial loading
- Verify checkpoint was saved correctly

---

## 📚 Additional Resources

- **Base Model API**: See `models/base_model.py`
- **Existing Examples**: Check `models/magic_wrapper.py`, `models/kairos_wrapper.py`
- **Dataset Handling**: See `data/dataset.py`
- **Metrics**: See `utils/metrics.py`

---

## ✅ Checklist for Model Integration

- [ ] Standalone implementation created in `models/implementations/your_model/`
- [ ] Wrapper class created inheriting from `BasePIDSModel`
- [ ] Model registered with `@ModelRegistry.register('your_model')`
- [ ] All mandatory methods implemented
- [ ] Configuration file created (optional)
- [ ] Unit tests written and passing
- [ ] Integration test with evaluate.py successful
- [ ] Documentation added (docstrings)
- [ ] Pretrained weights prepared (optional)

---

**Congratulations! You've successfully integrated a new PIDS model into the framework! 🎉**

Your model is now available for evaluation alongside other state-of-the-art approaches.
