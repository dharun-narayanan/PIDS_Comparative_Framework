"""
Pipeline builder for creating task-based execution pipelines.

This module provides utilities to build model-specific pipelines
based on YAML configuration files.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from .task_manager import TaskManager
from .task_registry import TaskRegistry
from models.model_builder import ModelBuilder

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Builds task-based pipelines for PIDS models.
    
    Each model can have a custom pipeline configuration that defines:
    - Which tasks to execute
    - Task-specific configurations
    - Task dependencies
    - Output paths
    """
    
    # Define standard task dependencies
    TASK_DEPENDENCIES = {
        'load_preprocessed_data': [],
        'construct_time_windows': ['load_preprocessed_data'],
        'graph_transformation': ['construct_time_windows'],
        'feature_extraction': ['graph_transformation'],
        'featurization_inference': ['feature_extraction'],
        'batch_construction': ['featurization_inference'],
        'model_inference': ['batch_construction'],
        'calculate_metrics': ['model_inference'],
        'attack_tracing': ['model_inference', 'calculate_metrics'],
    }
    
    # Map task names to registry functions
    TASK_FUNCTIONS = {
        'load_preprocessed_data': TaskRegistry.load_preprocessed_data,
        'construct_time_windows': TaskRegistry.construct_time_windows,
        'graph_transformation': TaskRegistry.graph_transformation,
        'feature_extraction': TaskRegistry.feature_extraction,
        'featurization_inference': TaskRegistry.featurization_inference,
        'batch_construction': TaskRegistry.batch_construction,
        'model_inference': TaskRegistry.model_inference,
        'calculate_metrics': TaskRegistry.calculate_metrics,
        'attack_tracing': TaskRegistry.attack_tracing,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline builder.
        
        Args:
            config: Global configuration dictionary
        """
        self.config = config
        self.task_manager = None
        self.model_builder = ModelBuilder(config_dir="configs/models")
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'PipelineBuilder':
        """
        Create pipeline builder from YAML config file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            PipelineBuilder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(config)
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Load model-specific configuration from configs/models/{model_name}.yaml
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.model_builder.load_config(model_name)
    
    def build_pipeline(
        self,
        model_name: str,
        tasks: Optional[List[str]] = None,
        force_restart: bool = False
    ) -> TaskManager:
        """
        Build a task pipeline for a specific model.
        
        Args:
            model_name: Name of the model (magic, kairos, etc.)
            tasks: List of tasks to include. If None, uses all standard tasks.
            force_restart: Force re-execution of all tasks
            
        Returns:
            Configured TaskManager
        """
        logger.info(f"Building pipeline for model: {model_name}")
        
        # Load model-specific configuration
        model_config = self.load_model_config(model_name)
        logger.info(f"Loaded model config: {model_config.get('name', model_name)}")
        
        # Create task manager
        self.task_manager = TaskManager(self.config, force_restart=force_restart)
        
        # Make a copy of dependencies that we can modify
        task_dependencies = self.TASK_DEPENDENCIES.copy()
        
        # Determine which tasks to include
        if tasks is None:
            # Use default pipeline tasks
            tasks = [
                'load_preprocessed_data',
                'construct_time_windows',
                'graph_transformation',
                'feature_extraction',
                'featurization_inference',
                'batch_construction',
                'model_inference',
                'calculate_metrics',
            ]
            
            # Check if model config specifies to skip time windows
            data_config = model_config.get('data', {})
            window_config = data_config.get('window', {})
            if window_config.get('size') is None:
                # Skip time window construction
                tasks.remove('construct_time_windows')
                # Update graph_transformation to depend on load_preprocessed_data instead
                task_dependencies['graph_transformation'] = ['load_preprocessed_data']
                logger.info(f"Skipping time windowing for {model_name}")
        
        logger.info(f"Pipeline tasks: {tasks}")
        
        # Register each task
        artifact_dir = Path(self.config.get('artifact_dir', 'artifacts')) / model_name
        
        for task_name in tasks:
            if task_name not in self.TASK_FUNCTIONS:
                logger.warning(f"Unknown task: {task_name}, skipping")
                continue
            
            # Get task-specific config with model config merged
            task_config = self._get_task_config(model_name, task_name, model_config)
            
            # Get dependencies (use modified dependencies if time windows skipped)
            dependencies = task_dependencies.get(task_name, [])
            
            # Set output path
            output_path = artifact_dir / task_name / 'output.pkl'
            
            # Register task
            self.task_manager.register_task(
                name=task_name,
                function=self.TASK_FUNCTIONS[task_name],
                dependencies=dependencies,
                config=task_config,
                output_path=output_path
            )
        
        logger.info(f"Registered {len(tasks)} tasks")
        
        return self.task_manager
    
    def _get_task_config(self, model_name: str, task_name: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific task.
        
        Merges global config, model config, and task-specific config.
        
        Args:
            model_name: Name of the model
            task_name: Name of the task
            model_config: Model configuration dictionary (loaded from configs/models/)
            
        Returns:
            Task configuration dictionary
        """
        # Start with empty config
        task_config = {}
        
        # Load model config if not provided
        if model_config is None:
            model_config = self.load_model_config(model_name)
        
        # Add model-specific data settings
        data_config = model_config.get('data', {})
        task_config['data_config'] = data_config
        
        # Add task-specific settings from model config
        if task_name == 'construct_time_windows':
            window_config = data_config.get('window', {})
            task_config.update(window_config)
        
        elif task_name == 'graph_transformation':
            graph_config = data_config.get('graph', {})
            task_config.update(graph_config)
        
        elif task_name == 'feature_extraction':
            features_config = data_config.get('features', {})
            task_config.update(features_config)
        
        elif task_name == 'model_inference':
            # Model inference needs checkpoint path and model builder
            checkpoint_config = model_config.get('checkpoint', {}).get('pretrained', {})
            task_config['checkpoint_config'] = checkpoint_config
            task_config['model_config'] = model_config
            task_config['model_builder'] = self.model_builder
            
            # Dataset name from global config
            dataset_name = self.config.get('data', {}).get('dataset', 'custom_soc')
            task_config['dataset_name'] = dataset_name
            
        elif task_name == 'calculate_metrics':
            eval_config = model_config.get('evaluation', {})
            task_config['metrics'] = eval_config.get('metrics', [])
        
        elif task_name == 'attack_tracing':
            tracing_config = model_config.get('evaluation', {}).get('tracing', {})
            task_config.update(tracing_config)
        
        # Add common settings
        task_config['model_name'] = model_name
        
        # Add global task overrides if present
        if 'tasks' in self.config and task_name in self.config['tasks']:
            task_config.update(self.config['tasks'][task_name])
        
        return task_config
    
    def build_and_execute(
        self,
        model_name: str,
        tasks: Optional[List[str]] = None,
        force_restart: bool = False
    ) -> Dict[str, Any]:
        """
        Build and execute a complete pipeline.
        
        Args:
            model_name: Name of the model
            tasks: List of tasks to execute
            force_restart: Force re-execution
            
        Returns:
            Dictionary of task results
        """
        task_manager = self.build_pipeline(model_name, tasks, force_restart)
        results = task_manager.execute_pipeline(tasks)
        
        # Save execution metadata
        artifact_dir = Path(self.config.get('artifact_dir', 'artifacts')) / model_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = artifact_dir / 'execution_metadata.json'
        task_manager.save_execution_metadata(metadata_path)
        
        return results
    
    @staticmethod
    def get_default_pipeline_config(model_name: str) -> Dict[str, Any]:
        """
        Get default pipeline configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Default pipeline configuration
        """
        # Base configuration common to all models
        base_config = {
            'tasks': [
                'load_preprocessed_data',
                'construct_time_windows',
                'graph_transformation',
                'feature_extraction',
                'featurization_inference',
                'batch_construction',
                'model_inference',
                'calculate_metrics',
            ],
            'task_configs': {
                'construct_time_windows': {
                    'window_size': 3600,  # 1 hour
                    'overlap': 0.0,
                },
                'graph_transformation': {
                    'type': 'none',
                },
                'feature_extraction': {
                    'method': 'one_hot',
                    'node_feat_dim': 128,
                    'edge_feat_dim': 64,
                },
                'batch_construction': {
                    'batch_size': 1,
                },
            }
        }
        
        # Model-specific configurations
        model_configs = {
            'magic': {
                'task_configs': {
                    'feature_extraction': {
                        'method': 'one_hot',
                        'node_feat_dim': 128,
                        'edge_feat_dim': 64,
                    },
                    'graph_transformation': {
                        'type': 'none',
                    },
                }
            },
            'kairos': {
                'task_configs': {
                    'feature_extraction': {
                        'method': 'one_hot',
                        'node_feat_dim': 100,
                        'edge_feat_dim': 100,
                    },
                    'construct_time_windows': {
                        'window_size': 600,  # 10 minutes
                        'overlap': 0.0,
                    },
                }
            },
            'orthrus': {
                'task_configs': {
                    'feature_extraction': {
                        'method': 'one_hot',
                        'node_feat_dim': 100,
                        'edge_feat_dim': 100,
                    },
                }
            },
            'threatrace': {
                'task_configs': {
                    'feature_extraction': {
                        'method': 'random',
                        'node_feat_dim': 128,
                        'edge_feat_dim': 64,
                    },
                }
            },
            'continuum_fl': {
                'task_configs': {
                    'feature_extraction': {
                        'method': 'one_hot',
                        'node_feat_dim': 128,
                        'edge_feat_dim': 64,
                    },
                }
            },
        }
        
        # Merge base config with model-specific config
        config = base_config.copy()
        if model_name in model_configs:
            # Deep merge task_configs
            if 'task_configs' in model_configs[model_name]:
                for task, task_conf in model_configs[model_name]['task_configs'].items():
                    if task in config['task_configs']:
                        config['task_configs'][task].update(task_conf)
                    else:
                        config['task_configs'][task] = task_conf
        
        return {'pipeline': config}


def create_pipeline_config_file(
    model_name: str,
    output_path: Path,
    custom_config: Optional[Dict[str, Any]] = None
):
    """
    Create a YAML configuration file for a model pipeline.
    
    Args:
        model_name: Name of the model
        output_path: Path to save YAML file
        custom_config: Optional custom configuration to merge
    """
    config = PipelineBuilder.get_default_pipeline_config(model_name)
    
    if custom_config:
        # Deep merge custom config
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, custom_config)
    
    # Save to YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created pipeline config: {output_path}")
