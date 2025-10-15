# PIDS Comparative Framework - Architecture and Extensibility Guide

## Overview

The PIDS Comparative Framework is designed as a **standalone, self-contained system** for evaluating and comparing state-of-the-art Provenance-based Intrusion Detection Systems (PIDS). The framework does NOT depend on external model repositories being present - all model implementations are self-contained within the framework.

## Architecture Principles

### 1. **Standalone Implementation**
- All model implementations are contained in `models/implementations/`
- No dependencies on external repositories or folders
- Each model has its own self-contained subpackage

### 2. **Plugin Architecture**
- Models register themselves via `@ModelRegistry.register()` decorator
- Easy to add new models without modifying core framework code
- Wrapper classes adapt standalone implementations to framework interface

### 3. **Unified Interface**
- All models inherit from `BasePIDSModel`
- Consistent API across all implementations
- Standard training, evaluation, and checkpoint management

## Directory Structure

```
PIDS_Comparative_Framework/
├── models/
│   ├── base_model.py              # Base class and registry
│   ├── implementations/           # Standalone model implementations
│   │   ├── magic/                # MAGIC implementation
│   │   │   ├── __init__.py
│   │   │   ├── gat.py            # Graph Attention Network
│   │   │   ├── autoencoder.py    # Masked autoencoder
│   │   │   ├── loss_func.py      # Loss functions
│   │   │   ├── eval.py           # Evaluation utilities
│   │   │   └── utils.py          # Helper functions
│   │   ├── kairos/               # Kairos implementation
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # Core model
│   │   │   ├── time_encoder.py   # Temporal encoding
│   │   │   └── utils.py
│   │   ├── orthrus/              # Orthrus implementation
│   │   ├── continuum_fl/         # Federated learning implementation
│   │   ├── threatrace/           # ThreaTrace implementation
│   │   └── utils/                # Shared utilities
│   │       ├── graph_utils.py
│   │       ├── temporal_utils.py
│   │       └── eval_utils.py
│   ├── magic_wrapper.py          # MAGIC adapter to framework
│   ├── kairos_wrapper.py         # Kairos adapter
│   ├── orthrus_wrapper.py        # Orthrus adapter
│   ├── continuum_fl_wrapper.py   # Continuum_FL adapter
│   └── threatrace_wrapper.py     # ThreaTrace adapter
├── data/
│   ├── dataset.py                # Dataset base classes
│   └── preprocessing/            # Data preprocessing utilities
├── experiments/
│   ├── evaluate.py               # Evaluation script
│   ├── train.py                  # Training script
│   └── compare.py                # Comparison script
└── utils/
    └── common.py                 # Common utilities
```

## Adding a New PIDS Model

To add a new state-of-the-art PIDS model to the framework, follow these steps:

### Step 1: Create Standalone Implementation

Create a new directory under `models/implementations/your_model/`:

```python
# models/implementations/your_model/__init__.py
"""
YourModel Implementation
Standalone implementation adapted for PIDS Comparative Framework.

Paper: "Your Paper Title"
Conference/Journal Year
"""

from .model import YourModelCore
from .utils import your_helper_functions

__all__ = ['YourModelCore', 'your_helper_functions']
```

### Step 2: Implement Core Model Components

```python
# models/implementations/your_model/model.py
import torch
import torch.nn as nn

class YourModelCore(nn.Module):
    """
    Core implementation of your model.
    
    This should be a self-contained implementation that doesn't
    depend on external repositories.
    """
    
    def __init__(self, config):
        super().__init__()
        # Initialize your model layers
        self.encoder = ...
        self.decoder = ...
    
    def forward(self, x):
        # Implement forward pass
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output
    
    def get_embeddings(self, x):
        """Extract embeddings (optional but recommended)."""
        return self.encoder(x)
```

### Step 3: Create Framework Wrapper

```python
# models/your_model_wrapper.py
"""
YourModel Wrapper for PIDS Comparative Framework
"""

import torch
from typing import Dict, Any
from pathlib import Path

from models.base_model import BasePIDSModel, ModelRegistry
from models.implementations.your_model import YourModelCore


@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    """
    YourModel: Brief description
    Paper: Conference/Journal Year
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize from standalone implementation
        self.model = YourModelCore(config)
        self.logger.info(f"YourModel initialized with {self.count_parameters()} parameters")
    
    def forward(self, batch):
        """Forward pass through model."""
        return self.model(batch)
    
    def train_epoch(self, dataloader, optimizer, **kwargs):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            output = self.forward(batch)
            loss = self.compute_loss(output, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def evaluate(self, dataloader, **kwargs):
        """Evaluate model."""
        self.model.eval()
        # Implement evaluation logic
        # Return Dict[str, float] with metrics
        return {'auc_roc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path, **kwargs):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded from {path}")
```

### Step 4: Register Model Configurations

Add your model configuration to `configs/models/your_model.yaml`:

```yaml
model:
  name: your_model
  architecture:
    hidden_dim: 256
    num_layers: 4
    dropout: 0.1
    # Add your model-specific parameters

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  
evaluation:
  metrics:
    - auc_roc
    - f1
    - precision
    - recall
```

### Step 5: Test Your Model

```bash
# Test on a single dataset
python experiments/evaluate.py \
    --model your_model \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --checkpoint checkpoints/your_model/checkpoint.pt

# Run complete evaluation
./scripts/run_evaluation.sh \
    --model your_model \
    --data-path data/custom_soc
```

## Best Practices

### 1. **Self-Contained Implementations**
- Copy and adapt code from original repositories
- Don't use `sys.path.insert()` or add external paths
- Include all necessary utilities in your implementation

### 2. **Minimal Dependencies**
- Use standard PyTorch operations when possible
- For graph operations, use PyTorch Geometric or DGL (already in requirements)
- Document any additional dependencies in your model's `__init__.py`

### 3. **Consistent Interface**
- Always inherit from `BasePIDSModel`
- Implement all required methods: `forward()`, `train_epoch()`, `evaluate()`, `save_checkpoint()`, `load_checkpoint()`
- Return Dict[str, float] from `evaluate()`

### 4. **Documentation**
- Include paper citation in docstrings
- Document all configuration parameters
- Provide usage examples

### 5. **Testing**
- Test with synthetic data first
- Verify checkpoint save/load works
- Ensure evaluation metrics are computed correctly

## Model Registry

The `ModelRegistry` class provides automatic model discovery and instantiation:

```python
# List all available models
from models.base_model import ModelRegistry
available_models = ModelRegistry.list_models()
print(f"Available models: {available_models}")

# Get a model instance
config = {'model_name': 'your_model', ...}
model = ModelRegistry.get_model('your_model', config)

# Check if a model is registered
if ModelRegistry.is_registered('your_model'):
    model = ModelRegistry.get_model('your_model', config)
```

## Configuration System

Models can be configured via:

1. **YAML files** in `configs/models/`
2. **Command-line arguments** (override YAML)
3. **Python dictionaries** (programmatic usage)

Example:
```python
from utils.config import load_config

# Load base config
config = load_config('configs/models/your_model.yaml')

# Override specific parameters
config.update({
    'hidden_dim': 512,
    'learning_rate': 0.0001
})

# Create model
model = ModelRegistry.get_model('your_model', config)
```

## Evaluation Pipeline

The framework provides a unified evaluation pipeline:

```python
from experiments.evaluate import evaluate_model

results = evaluate_model(
    model=model,
    test_loader=test_loader,
    device='cuda',
    metrics=['auc_roc', 'f1', 'precision', 'recall']
)

print(f"Results: {results}")
```

## Extending Data Loading

To support new data formats:

```python
# data/your_dataset.py
from data.dataset import BasePIDSDataset

class YourDataset(BasePIDSDataset):
    """Custom dataset implementation."""
    
    def load_data(self):
        # Load your data format
        pass
    
    def __getitem__(self, idx):
        # Return data and label
        return self.data[idx], self.labels[idx]
```

## Future Model Integration

When a new state-of-the-art PIDS model is published:

1. **Review the paper** - understand architecture and requirements
2. **Extract core components** - identify essential model components
3. **Create standalone implementation** - adapt code to framework
4. **Write wrapper** - integrate with `BasePIDSModel` interface
5. **Add configuration** - create YAML config file
6. **Test thoroughly** - verify training and evaluation
7. **Document** - update this guide with any model-specific notes

## Troubleshooting

### Import Errors
- Ensure all imports are from `models.implementations.your_model`
- Don't use external path manipulation
- Check that `__init__.py` exports all necessary components

### Shape Mismatches
- Verify input/output dimensions match expected format
- Check batch handling in `forward()` and `train_epoch()`
- Ensure compatibility with framework's data loaders

### Training Issues
- Verify loss computation is correct
- Check optimizer configuration
- Monitor gradients for vanishing/exploding issues

### Evaluation Errors
- Ensure `evaluate()` returns Dict[str, float]
- Verify metrics are computed correctly
- Check for NaN or inf values in outputs

## Contact and Support

For questions or issues with adding new models:
1. Check this documentation
2. Review existing model implementations for examples
3. Open an issue with framework maintainers

## Version History

- **v1.0** - Initial standalone framework
- **v1.1** - Added MAGIC, Kairos, Orthrus, Continuum_FL, ThreaTrace
- **Future** - Your model here!
