"""
Base class for all PIDS models in the framework.
All models should inherit from this class and implement required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BasePIDSModel(ABC, nn.Module):
    """
    Abstract base class for PIDS models.
    
    All models must implement:
    - forward(): Forward pass through the model
    - train_epoch(): Training logic for one epoch
    - evaluate(): Evaluation logic
    - save_checkpoint(): Save model state
    - load_checkpoint(): Load model state
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(BasePIDSModel, self).__init__()
        self.config = config
        self.model_name = config.get('model_name', 'base_model')
        
        # Safe device handling
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.device = torch.device(device_str)
        except RuntimeError:
            # Fallback to CPU if device string is invalid
            self.device = torch.device('cpu')
            self.logger = logging.getLogger(f"{__name__}.{self.model_name}")
            self.logger.warning(f"Invalid device '{device_str}', falling back to CPU")
        
        self.logger = logging.getLogger(f"{__name__}.{self.model_name}")
        
    @abstractmethod
    def forward(self, batch: Any) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch (format depends on model)
            
        Returns:
            Model output (predictions, embeddings, etc.)
        """
        pass
    
    @abstractmethod
    def train_epoch(self, dataloader, optimizer, **kwargs) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics (loss, etc.)
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataloader, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation data loader
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary containing evaluation metrics (AUC, F1, etc.)
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: Path, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path, **kwargs) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            **kwargs: Additional loading arguments
        """
        pass
    
    def get_embeddings(self, batch: Any) -> torch.Tensor:
        """
        Extract embeddings from the model (optional).
        
        Args:
            batch: Input batch
            
        Returns:
            Embeddings tensor
        """
        raise NotImplementedError(f"{self.model_name} does not implement get_embeddings()")
    
    def to_device(self, device: Optional[str] = None) -> 'BasePIDSModel':
        """Move model to specified device."""
        device = device or self.device
        self.device = device
        return self.to(device)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'num_parameters': self.count_parameters(),
            'device': self.device,
            'config': self.config
        }
    
    def set_train_mode(self) -> None:
        """Set model to training mode."""
        self.train()
        
    def set_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.eval()
        
    @staticmethod
    def load_pretrained(checkpoint_path: Path, config: Dict[str, Any]) -> 'BasePIDSModel':
        """
        Load a pretrained model.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
            config: Model configuration
            
        Returns:
            Loaded model instance
        """
        raise NotImplementedError("Subclass must implement load_pretrained()")


class ModelRegistry:
    """Registry for all available models."""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str, config: Dict[str, Any]) -> BasePIDSModel:
        """Get a model instance by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry. "
                           f"Available models: {list(cls._models.keys())}")
        return cls._models[name](config)
    
    @classmethod
    def list_models(cls):
        """List all registered models."""
        return list(cls._models.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered."""
        return name in cls._models
