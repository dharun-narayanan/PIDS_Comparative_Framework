"""
Model Builder - Constructs models directly from configuration files.

This module replaces model-specific wrapper files by building models
dynamically from shared encoder and decoder components based on YAML configs.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

from models.shared_encoders import get_encoder, MultiEncoder
from models.shared_decoders import get_decoder


logger = logging.getLogger(__name__)


class GenericModel(nn.Module):
    """
    Generic model class that works with any encoder-decoder combination.
    
    This replaces all model-specific wrapper classes (MAGIC, Kairos, etc.)
    by dynamically constructing models from shared components.
    """
    
    def __init__(
        self,
        encoders: Union[nn.Module, List[nn.Module]],
        decoders: Union[nn.Module, Dict[str, nn.Module]],
        config: Dict[str, Any]
    ):
        """
        Initialize generic model.
        
        Args:
            encoders: Single encoder or list of encoders
            decoders: Single decoder or dict of named decoders
            config: Model configuration dictionary
        """
        super(GenericModel, self).__init__()
        
        self.config = config
        self.model_name = config.get('name', 'unknown')
        
        # Setup encoders
        if isinstance(encoders, list):
            self.multi_encoder = True
            self.encoders = nn.ModuleList(encoders)
            
            # Setup encoder combination
            combination_config = config.get('architecture', {}).get('encoder_combination', {})
            self.encoder_combination_method = combination_config.get('method', 'concat')
        else:
            self.multi_encoder = False
            self.encoder = encoders
        
        # Setup decoders
        if isinstance(decoders, dict):
            self.multi_decoder = True
            self.decoders = nn.ModuleDict(decoders)
            self.primary_decoder_key = 'primary' if 'primary' in decoders else list(decoders.keys())[0]
        else:
            self.multi_decoder = False
            self.decoder = decoders
    
    def encode(self, data):
        """
        Encode input data using encoder(s).
        
        Args:
            data: PyTorch Geometric Data object or dict
            
        Returns:
            Node embeddings [num_nodes, embed_dim]
        """
        if self.multi_encoder:
            # Get embeddings from each encoder
            embeddings = []
            for encoder in self.encoders:
                h = encoder(data.x, data.edge_index, 
                           edge_attr=getattr(data, 'edge_attr', None))
                embeddings.append(h)
            
            # Combine embeddings
            if self.encoder_combination_method == 'concat':
                return torch.cat(embeddings, dim=-1)
            elif self.encoder_combination_method == 'mean':
                return torch.stack(embeddings).mean(dim=0)
            elif self.encoder_combination_method == 'max':
                return torch.stack(embeddings).max(dim=0)[0]
            else:
                raise ValueError(f"Unknown combination method: {self.encoder_combination_method}")
        else:
            return self.encoder(data.x, data.edge_index,
                              edge_attr=getattr(data, 'edge_attr', None))
    
    def decode(self, h, data, decoder_name=None, inference=False):
        """
        Decode embeddings using decoder(s).
        
        Args:
            h: Node embeddings [num_nodes, embed_dim]
            data: PyTorch Geometric Data object
            decoder_name: Specific decoder to use (for multi-decoder)
            inference: Whether in inference mode
            
        Returns:
            Predictions or loss
        """
        if self.multi_decoder:
            decoder_name = decoder_name or self.primary_decoder_key
            decoder = self.decoders[decoder_name]
        else:
            decoder = self.decoder
        
        # Get source and destination embeddings for edge-level decoders
        if hasattr(decoder, 'forward'):
            sig = decoder.forward.__code__.co_varnames
            
            if 'h_src' in sig and 'h_dst' in sig:
                # Edge decoder
                h_src = h[data.edge_index[0]]
                h_dst = h[data.edge_index[1]]
                
                edge_features = getattr(data, 'edge_attr', None)
                edge_labels = getattr(data, 'y', None)
                
                return decoder(h_src, h_dst, edge_features, edge_labels, inference)
            
            elif 'h_current' in sig:
                # Contrastive decoder
                h_previous = getattr(data, 'h_previous', None)
                return decoder(h, h_previous, inference=inference)
            
            else:
                # Node-level decoder or reconstruction
                labels = getattr(data, 'y', None)
                features = getattr(data, 'x', None)
                
                if 'target_features' in sig:
                    return decoder(h, features, inference)
                else:
                    return decoder(h, labels, inference=inference)
    
    def forward(self, data, decoder_name=None, inference=False):
        """
        Full forward pass through model.
        
        Args:
            data: PyTorch Geometric Data object
            decoder_name: Specific decoder to use (for multi-decoder)
            inference: Whether in inference mode
            
        Returns:
            Predictions or loss
        """
        # Encode
        h = self.encode(data)
        
        # Decode
        if self.multi_decoder and not decoder_name:
            # Return outputs from all decoders
            outputs = {}
            for name in self.decoders.keys():
                outputs[name] = self.decode(h, data, name, inference)
            return outputs
        else:
            return self.decode(h, data, decoder_name, inference)
    
    def get_embeddings(self, data):
        """Get node embeddings without decoding."""
        return self.encode(data)


class ModelBuilder:
    """
    Factory class for building models from configuration files.
    """
    
    def __init__(self, config_dir: str = "configs/models"):
        """
        Initialize ModelBuilder.
        
        Args:
            config_dir: Directory containing model configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs_cache = {}
    
    def load_config(self, model_name: str) -> Dict[str, Any]:
        """
        Load model configuration from YAML file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Configuration dictionary
        """
        if model_name in self.configs_cache:
            return self.configs_cache[model_name]
        
        config_path = self.config_dir / f"{model_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.configs_cache[model_name] = config
        logger.info(f"Loaded configuration for {model_name}")
        
        return config
    
    def build_encoder(self, encoder_config: Dict[str, Any]) -> nn.Module:
        """
        Build encoder from configuration.
        
        Args:
            encoder_config: Encoder configuration dict
            
        Returns:
            Encoder module
        """
        encoder_type = encoder_config.pop('type')
        name = encoder_config.pop('name', None)  # Remove name if present
        
        # Normalize parameter names: map *_dim to *_channels
        param_mapping = {
            'in_dim': 'in_channels',
            'hidden_dim': 'hidden_channels',
            'out_dim': 'out_channels'
        }
        for old_key, new_key in param_mapping.items():
            if old_key in encoder_config and new_key not in encoder_config:
                encoder_config[new_key] = encoder_config.pop(old_key)
        
        # Remove model-specific parameters not supported by base encoders
        unsupported_params = [
            'use_edge_attr', 'time_encoding', 'time_encoder', 'time_dim',
            'max_time_delta', 'memory_dim', 'max_nodes', 'memory_updater',
            'msg_dim', 'aggregator', 'normalization', 'add_self_loops',
            'temporal', 'n_snapshot', 'pooling', 'negative_slope', 'use_all_hidden',
            'loss_fn', 'alpha_l', 'use_graphchi', 'graphchi_niters', 'graphchi_nshards'
        ]
        for param in unsupported_params:
            encoder_config.pop(param, None)
        
        encoder = get_encoder(encoder_type, encoder_config)
        
        # Restore config values
        encoder_config['type'] = encoder_type
        if name:
            encoder_config['name'] = name
        
        return encoder
    
    def build_decoder(self, decoder_config: Dict[str, Any]) -> nn.Module:
        """
        Build decoder from configuration.
        
        Args:
            decoder_config: Decoder configuration dict
            
        Returns:
            Decoder module
        """
        decoder_type = decoder_config.pop('type')
        
        # Remove model-specific parameters not supported by base decoders
        unsupported_params = [
            'use_edge_features', 'mask_rate', 'loss_fn', 'alpha_l',
            'norm', 'use_all_hidden', 'temperature', 'activation'
        ]
        for param in unsupported_params:
            decoder_config.pop(param, None)
        
        decoder = get_decoder(decoder_type, decoder_config)
        
        # Restore config
        decoder_config['type'] = decoder_type
        
        return decoder
    
    def build_model(
        self,
        model_name: str,
        override_config: Optional[Dict[str, Any]] = None
    ) -> GenericModel:
        """
        Build model from configuration file.
        
        Args:
            model_name: Name of the model (corresponds to config file)
            override_config: Optional config overrides
            
        Returns:
            GenericModel instance
        """
        # Load configuration
        config = self.load_config(model_name)
        
        # Apply overrides
        if override_config:
            config = self._merge_configs(config, override_config)
        
        arch_config = config.get('architecture', {})
        
        # Build encoder(s)
        if 'encoders' in arch_config:
            # Multi-encoder model
            encoders = []
            for enc_cfg in arch_config['encoders']:
                enc_cfg_copy = enc_cfg.copy()
                encoders.append(self.build_encoder(enc_cfg_copy))
            encoder = encoders
        elif 'encoder' in arch_config:
            # Single encoder model
            enc_cfg = arch_config['encoder'].copy()
            encoder = self.build_encoder(enc_cfg)
        else:
            raise ValueError(f"No encoder configuration found for {model_name}")
        
        # Build decoder(s)
        if 'decoders' in arch_config:
            # Multi-decoder model
            decoders = {}
            for name, dec_cfg in arch_config['decoders'].items():
                dec_cfg_copy = dec_cfg.copy()
                decoders[name] = self.build_decoder(dec_cfg_copy)
            decoder = decoders
        elif 'decoder' in arch_config:
            # Single decoder model
            dec_cfg = arch_config['decoder'].copy()
            
            # Handle nested decoder config (e.g., decoder.edge)
            if not isinstance(dec_cfg, dict):
                raise ValueError(f"Invalid decoder config for {model_name}")
            
            # Check if decoder config has nested structure
            if 'type' not in dec_cfg:
                # Nested structure, take first decoder
                first_key = list(dec_cfg.keys())[0]
                dec_cfg = dec_cfg[first_key].copy()
            
            decoder = self.build_decoder(dec_cfg)
        else:
            raise ValueError(f"No decoder configuration found for {model_name}")
        
        # Build model
        model = GenericModel(encoder, decoder, config)
        
        logger.info(f"Built model: {model_name}")
        logger.info(f"  Encoders: {len(encoder) if isinstance(encoder, list) else 1}")
        logger.info(f"  Decoders: {len(decoder) if isinstance(decoder, dict) else 1}")
        
        return model
    
    def load_pretrained(
        self,
        model: GenericModel,
        checkpoint_path: str,
        strict: bool = False,
        device: str = 'cpu'
    ) -> GenericModel:
        """
        Load pretrained weights into model.
        
        Args:
            model: Model instance
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
            device: Device to load model on
            
        Returns:
            Model with loaded weights
        """
        checkpoint_file = Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_file}")
            return model
        
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        
        try:
            checkpoint = torch.load(str(checkpoint_file), map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
            
            logger.info(f"Successfully loaded checkpoint")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            if strict:
                raise
        
        return model
    
    def build_and_load(
        self,
        model_name: str,
        dataset_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        override_config: Optional[Dict[str, Any]] = None
    ) -> GenericModel:
        """
        Build model and optionally load pretrained weights.
        
        Args:
            model_name: Name of the model
            dataset_name: Dataset name for finding checkpoint
            checkpoint_path: Explicit checkpoint path (overrides dataset-based path)
            device: Device to load model on
            override_config: Optional config overrides
            
        Returns:
            Model instance with optional pretrained weights
        """
        # Build model
        model = self.build_model(model_name, override_config)
        model = model.to(device)
        
        # Load checkpoint if specified
        config = self.configs_cache[model_name]
        checkpoint_config = config.get('checkpoint', {}).get('pretrained', {})
        
        if checkpoint_config.get('enabled', True):
            if checkpoint_path is None and dataset_name:
                # Build checkpoint path from config template
                checkpoint_template = checkpoint_config.get('path', '')
                checkpoint_path = checkpoint_template.format(
                    model_name=model_name,
                    dataset=dataset_name
                )
            
            if checkpoint_path:
                strict = checkpoint_config.get('strict', False)
                model = self.load_pretrained(model, checkpoint_path, strict, device)
        
        return model
    
    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ModelBuilder._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def list_available_models(self) -> List[str]:
        """List all available model configurations."""
        models = []
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.stem != "template":
                models.append(config_file.stem)
        return sorted(models)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_model(
    model_name: str,
    dataset_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    config_dir: str = "configs/models"
) -> GenericModel:
    """
    Convenience function to create a model.
    
    Args:
        model_name: Name of the model
        dataset_name: Dataset name for finding checkpoint
        checkpoint_path: Explicit checkpoint path
        device: Device to load model on
        config_dir: Directory containing model configs
        
    Returns:
        Model instance
    """
    builder = ModelBuilder(config_dir)
    return builder.build_and_load(model_name, dataset_name, checkpoint_path, device)


def list_models(config_dir: str = "configs/models") -> List[str]:
    """
    List all available models.
    
    Args:
        config_dir: Directory containing model configs
        
    Returns:
        List of model names
    """
    builder = ModelBuilder(config_dir)
    return builder.list_available_models()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Available models:")
    for model in list_models():
        print(f"  - {model}")
    
    # Build a model
    print("\nBuilding MAGIC model...")
    model = create_model("magic", dataset_name="cadets", device="cpu")
    print(f"Model built successfully: {model.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
