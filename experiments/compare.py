#!/usr/bin/env python3
"""
Compare multiple PIDS models on the same dataset.

This script trains and evaluates multiple PIDS models, then generates
a comprehensive comparison report.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.common import setup_logging, load_config, set_seed
from utils.metrics import compute_detection_metrics
from models import ModelRegistry
from data.dataset import get_dataloader
import logging

setup_logging()
logger = logging.getLogger("compare_models")


class ModelComparator:
    """Compare multiple PIDS models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.training_times = {}
        self.inference_times = {}
    
    def run_comparison(self):
        """Run comparison experiment."""
        logger.info("="*80)
        logger.info("PIDS Model Comparison Experiment")
        logger.info("="*80)
        
        # Load dataset configuration
        dataset_config = load_config(self.config['dataset']['config'])
        
        # Get models to compare
        models_to_compare = self.config.get('models', [])
        
        logger.info(f"\nModels to compare: {len(models_to_compare)}")
        for model_info in models_to_compare:
            logger.info(f"  - {model_info['name']}")
        
        # Run each model
        for model_info in models_to_compare:
            model_name = model_info['name']
            model_config_path = model_info['config']
            variants = model_info.get('variants', [model_name])
            
            for variant in variants:
                logger.info(f"\n{'='*80}")
                logger.info(f"Running: {variant}")
                logger.info(f"{'='*80}\n")
                
                try:
                    results = self.run_single_model(
                        variant,
                        model_config_path,
                        dataset_config
                    )
                    self.results[variant] = results
                except Exception as e:
                    logger.error(f"Error running {variant}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Generate comparison report
        self.generate_report()
    
    def run_single_model(
        self,
        model_name: str,
        model_config_path: str,
        dataset_config: Dict
    ) -> Dict:
        """Run a single model."""
        # Load model configuration
        model_config = load_config(model_config_path)
        
        # Set seed for reproducibility
        seed = self.config.get('training', {}).get('seed', 42)
        set_seed(seed)
        
        # Create model
        logger.info(f"Creating model: {model_name}")
        model = ModelRegistry.create_model(model_name, model_config)
        
        # Load data
        logger.info("Loading dataset...")
        train_loader = get_dataloader(
            dataset_config,
            split='train',
            batch_size=model_config.get('training', {}).get('batch_size', 32)
        )
        val_loader = get_dataloader(
            dataset_config,
            split='val',
            batch_size=model_config.get('training', {}).get('batch_size', 32)
        )
        test_loader = get_dataloader(
            dataset_config,
            split='test',
            batch_size=model_config.get('training', {}).get('batch_size', 32)
        )
        
        # Training
        logger.info("Training model...")
        train_start = time.time()
        
        # Simple training loop (can be replaced with experiments/train.py)
        num_epochs = self.config.get('training', {}).get('max_epochs', 100)
        best_val_auroc = 0.0
        
        for epoch in range(num_epochs):
            # Training epoch
            train_metrics = model.train_epoch(train_loader, None)
            
            # Validation
            if (epoch + 1) % 5 == 0:
                val_metrics = model.evaluate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics.get('loss', 0):.4f} - "
                    f"Val AUROC: {val_metrics.get('auroc', 0):.4f}"
                )
                
                # Save best model
                if val_metrics.get('auroc', 0) > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
                    checkpoint_dir = Path(self.config['checkpointing']['save_dir']) / model_name
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_checkpoint(checkpoint_dir / 'best.pt')
        
        train_time = time.time() - train_start
        self.training_times[model_name] = train_time
        
        # Evaluation
        logger.info("Evaluating model...")
        
        # Load best checkpoint
        checkpoint_path = Path(self.config['checkpointing']['save_dir']) / model_name / 'best.pt'
        if checkpoint_path.exists():
            model.load_checkpoint(checkpoint_path)
        
        # Test evaluation
        inference_start = time.time()
        test_metrics = model.evaluate(test_loader)
        inference_time = time.time() - inference_start
        self.inference_times[model_name] = inference_time
        
        # Compile results
        results = {
            'metrics': test_metrics,
            'training_time': train_time,
            'inference_time': inference_time,
            'best_val_auroc': best_val_auroc
        }
        
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  AUROC: {test_metrics.get('auroc', 0):.4f}")
        logger.info(f"  AUPRC: {test_metrics.get('auprc', 0):.4f}")
        logger.info(f"  F1 Score: {test_metrics.get('f1_score', 0):.4f}")
        logger.info(f"  Training Time: {train_time:.2f}s")
        logger.info(f"  Inference Time: {inference_time:.2f}s")
        
        return results
    
    def generate_report(self):
        """Generate comparison report."""
        logger.info("\n" + "="*80)
        logger.info("Generating Comparison Report")
        logger.info("="*80 + "\n")
        
        if not self.results:
            logger.error("No results to report!")
            return
        
        # Create results directory
        results_dir = Path(self.config['results']['output_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile results DataFrame
        data = []
        for model_name, results in self.results.items():
            metrics = results['metrics']
            row = {
                'Model': model_name,
                'AUROC': metrics.get('auroc', 0),
                'AUPRC': metrics.get('auprc', 0),
                'F1 Score': metrics.get('f1_score', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'Accuracy': metrics.get('accuracy', 0),
                'Training Time (s)': results['training_time'],
                'Inference Time (s)': results['inference_time']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by AUROC
        df = df.sort_values('AUROC', ascending=False)
        
        # Print results table
        print("\n" + "="*80)
        print("Model Comparison Results")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        # Save as CSV
        csv_path = results_dir / 'comparison_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save as JSON
        json_path = results_dir / 'comparison_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to: {json_path}")
        
        # Generate plots
        if self.config.get('results', {}).get('generate_plots', True):
            self.generate_plots(df, results_dir)
    
    def generate_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate comparison plots."""
        logger.info("Generating plots...")
        
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 1. Metrics comparison bar chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = ['AUROC', 'AUPRC', 'F1 Score']
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            df_sorted = df.sort_values(metric, ascending=True)
            ax.barh(df_sorted['Model'], df_sorted[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for i, v in enumerate(df_sorted[metric]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: metrics_comparison.png")
        
        # 2. Performance vs Time scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df['Training Time (s)'],
            df['AUROC'],
            s=200,
            alpha=0.6,
            c=df['F1 Score'],
            cmap='viridis'
        )
        
        # Annotate points
        for idx, row in df.iterrows():
            ax.annotate(
                row['Model'],
                (row['Training Time (s)'], row['AUROC']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('AUROC')
        ax.set_title('Model Performance vs Training Time')
        plt.colorbar(scatter, label='F1 Score')
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: performance_vs_time.png")
        
        # 3. Radar chart for comprehensive comparison
        if len(df) > 1:
            self.plot_radar_chart(df, plots_dir)
    
    def plot_radar_chart(self, df: pd.DataFrame, output_dir: Path):
        """Create radar chart for model comparison."""
        import numpy as np
        
        # Select metrics for radar chart
        metrics = ['AUROC', 'AUPRC', 'F1 Score', 'Precision', 'Recall']
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        for idx, row in df.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.15)
        
        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Comprehensive Model Comparison', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: radar_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple PIDS models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=project_root / 'configs' / 'experiments' / 'compare_all.yaml',
        help='Experiment configuration file'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to compare (overrides config)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to use (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override models if specified
    if args.models:
        config['models'] = [{'name': m, 'config': f'configs/models/{m}.yaml'} 
                           for m in args.models]
    
    # Override dataset if specified
    if args.dataset:
        config['dataset']['name'] = args.dataset
        config['dataset']['config'] = f'configs/datasets/{args.dataset}.yaml'
    
    # Create comparator
    comparator = ModelComparator(config)
    
    # Run comparison
    comparator.run_comparison()
    
    logger.info("\n" + "="*80)
    logger.info("Comparison complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
