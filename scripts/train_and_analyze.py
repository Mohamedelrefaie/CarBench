#!/usr/bin/env python
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Model-Agnostic Training and Evaluation Script. Supports both single model evaluation and comprehensive 
training with proper normalization.

NORMALIZATION:
- All metrics are computed in ORIGINAL scale (Pascals) for interpretability
- Models internally handle normalization as needed for training stability
- The framework ensures consistent normalization/    # Initialize components
    logger.info(f"Initializing {args.model} for training...")
    wrapper = load_model_wrapper(args.model, config, logger)
    evaluator = UnifiedEvaluator(wrapper, save_dir=args.save_dir, enable_bootstrap=args.enable_bootstrap)
    
    # Setupalization flow
"""

import sys
import os
import argparse
import yaml
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from datetime import datetime
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from benchmark.evaluator import UnifiedEvaluator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config if config is not None else {}


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge override configuration into base configuration."""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_model_wrapper(model_name: str, config: Dict, logger):
    """Dynamically load the model wrapper."""
    wrapper_path = os.path.join(project_root, 'models', model_name, 'wrapper.py')
    
    if not os.path.exists(wrapper_path):
        raise FileNotFoundError(f"Wrapper not found for model {model_name} at {wrapper_path}")
    
    logger.info(f"Loading wrapper from {wrapper_path}")
    
    spec = importlib.util.spec_from_file_location(f"{model_name}_wrapper", wrapper_path)
    wrapper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper_module)
    
    # Find the wrapper class
    wrapper_class = None
    for name, obj in wrapper_module.__dict__.items():
        if name.endswith('Wrapper') and hasattr(obj, '__bases__'):
            wrapper_class = obj
            break

    if wrapper_class is None:
        raise ValueError(f"No wrapper class found in {wrapper_path}")

    return wrapper_class(config)


class ModelAgnosticAnalyzer:
    """Pure model-agnostic analysis - only uses actual model results"""
    
    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = Path(save_dir) / model_name.lower()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set publication-quality plotting
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.family': 'sans-serif'
        })
    
    def save_learning_curves(self, evaluator, model_name: str):
        """Save comprehensive learning curves"""
        self.logger.info("Generating learning curves...")
        
        curves_dir = self.save_dir / 'learning_curves'
        curves_dir.mkdir(exist_ok=True)
        
        # Extract training history from evaluator
        if hasattr(evaluator, 'training_history') and evaluator.training_history['train_loss']:
            history = evaluator.training_history
        else:
            self.logger.warning("No training history found")
            return
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Save raw data
        self._save_training_data(history, epochs, curves_dir)
        
        # Create visualizations
        self._create_learning_curve_plots(history, epochs, model_name, curves_dir)
        
        self.logger.info(f"Learning curves saved to {curves_dir}")
    
    def _save_training_data(self, history: Dict, epochs: List[int], curves_dir: Path):
        """Save training data in multiple formats"""
        # Prepare data
        df_data = {
            'epoch': epochs,
            'train_loss': history['train_loss'],
            'val_loss': history.get('val_loss', history['train_loss'])
        }
        
        if history.get('learning_rates'):
            df_data['learning_rate'] = history['learning_rates']
        
        # Save as CSV
        pd.DataFrame(df_data).to_csv(curves_dir / 'training_curves.csv', index=False)
        
        # Save as JSON with metadata
        metadata = {
            'model_name': self.model_name,
            'total_epochs': len(epochs),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history.get('val_loss', history['train_loss'])[-1],
            'best_epoch': np.argmin(history.get('val_loss', history['train_loss'])) + 1,
            'best_val_loss': min(history.get('val_loss', history['train_loss'])),
            'timestamp': datetime.now().isoformat()
        }
        
        data = {**history, 'epochs': epochs, 'metadata': metadata}

        def _np_default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(curves_dir / 'training_history.json', 'w') as f:
            json.dump(data, f, indent=2, default=_np_default)
    
    def _create_learning_curve_plots(self, history: Dict, epochs: List[int], 
                                   model_name: str, curves_dir: Path):
        """Create comprehensive learning curve visualization"""
        train_loss = history['train_loss']
        val_loss = history.get('val_loss', train_loss)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Loss curves (linear scale)
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Training', linewidth=2, alpha=0.8)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(val_loss)
        axes[0, 0].scatter(epochs[best_epoch], val_loss[best_epoch], 
                          color='gold', s=100, zorder=5, 
                          label=f'Best: {val_loss[best_epoch]:.6f}')
        
        # 2. Loss curves (log scale)
        axes[0, 1].semilogy(epochs, train_loss, 'b-', label='Training', linewidth=2, alpha=0.8)
        axes[0, 1].semilogy(epochs, val_loss, 'r-', label='Validation', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].set_title('Loss Curves (Log Scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, which='both')
        
        # 3. Learning rate schedule
        if history.get('learning_rates'):
            axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2, alpha=0.8)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No learning rate data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate Schedule')
        
        # 4. Generalization gap
        gap = np.array(val_loss) - np.array(train_loss)
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2, alpha=0.8)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(epochs, 0, gap, alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation - Training Loss')
        axes[1, 1].set_title('Generalization Gap')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        for fmt in ['png', 'pdf']:
            plt.savefig(curves_dir / f'training_analysis.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a simple, clean plot for papers
        self._create_paper_plot(epochs, train_loss, val_loss, model_name, curves_dir)
    
    def _create_paper_plot(self, epochs, train_loss, val_loss, model_name, curves_dir):
        """Create a clean plot suitable for papers"""
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2.5)
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2.5)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'{model_name} Training Progress', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(curves_dir / 'loss_curve_clean.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_metrics(self, results: Dict, evaluator, model_name: str):
        """Save comprehensive metrics and analysis"""
        self.logger.info("Saving comprehensive metrics...")
        
        metrics_dir = self.save_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        # Create summary table
        self._create_summary_table(results, metrics_dir)
        
        # Create detailed report
        self._create_detailed_report(results, metrics_dir)
        
        # Save bootstrap results if available
        if 'bootstrap_metrics' in results and results['bootstrap_metrics']:
            self._save_bootstrap_results(results['bootstrap_metrics'], metrics_dir)
        
        self.logger.info(f"Metrics saved to {metrics_dir}")
    
    def _save_bootstrap_results(self, bootstrap_metrics: Dict, metrics_dir: Path):
        """Save bootstrap confidence intervals to files."""
        self.logger.info("Saving bootstrap confidence intervals...")
        
        bootstrap_dir = metrics_dir / 'bootstrap'
        bootstrap_dir.mkdir(exist_ok=True)
        
        # Save as JSON (full detail with indices if available)
        json_path = bootstrap_dir / 'bootstrap_ci.json'
        with open(json_path, 'w') as f:
            json.dump(bootstrap_metrics, f, indent=2)
        
        # Save as CSV (summary table)
        csv_data = []
        for metric_name, result_dict in bootstrap_metrics.items():
            csv_data.append({
                'Metric': metric_name,
                'Mean': result_dict.get('mean', 'N/A'),
                'Std': result_dict.get('std', 'N/A'),
                'CI_Lower': result_dict.get('ci_lower', 'N/A'),
                'CI_Upper': result_dict.get('ci_upper', 'N/A'),
                'CI_Width': result_dict.get('ci_upper', 0) - result_dict.get('ci_lower', 0) if 'ci_upper' in result_dict else 'N/A'
            })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(bootstrap_dir / 'bootstrap_ci.csv', index=False)
        
        # Save formatted table as text
        with open(bootstrap_dir / 'bootstrap_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("BOOTSTRAP CONFIDENCE INTERVALS (B=2000, 95% CI)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Metric':<20} {'Mean':>12} {'CI Lower':>12} {'CI Upper':>12} {'Std':>12}\n")
            f.write("-"*80 + "\n")
            
            for metric_name, result_dict in bootstrap_metrics.items():
                mean = result_dict.get('mean', 0)
                ci_lower = result_dict.get('ci_lower', 0)
                ci_upper = result_dict.get('ci_upper', 0)
                std = result_dict.get('std', 0)
                
                f.write(f"{metric_name:<20} {mean:>12.6f} {ci_lower:>12.6f} {ci_upper:>12.6f} {std:>12.6f}\n")
        
        self.logger.info(f"✓ Bootstrap results saved to {bootstrap_dir}/")
        self.logger.info(f"  - JSON: {json_path.name}")
        self.logger.info(f"  - CSV: bootstrap_ci.csv")
        self.logger.info(f"  - Summary: bootstrap_summary.txt")
    
    def _create_summary_table(self, results: Dict, metrics_dir: Path):
        """Create summary table in multiple formats"""
        acc = results.get('accuracy_metrics', {})
        comp = results.get('computational_metrics', {})
        inf = results.get('inference_metrics', {})
        train = results.get('training_metrics', {})
        
        summary = {
            'Model': self.model_name,
            'Parameters': f"{comp.get('total_parameters', 0):,}",
            'GFLOPs': f"{comp.get('gflops', 0):.2f}",
            'Training Time (h)': f"{train.get('total_training_time_hours', 0):.2f}",
            'R² Score': f"{acc.get('r2', 0):.4f}",
            'MSE': f"{acc.get('mse', 0):.6f}",
            'MAE (Pa)': f"{acc.get('mae', 0):.2f}",
            'RMSE (Pa)': f"{acc.get('rmse', 0):.2f}",
            'Inference (ms)': f"{inf.get('mean_inference_time_ms', 0):.2f}"
        }
        
        # Save as CSV
        pd.DataFrame([summary]).to_csv(metrics_dir / 'summary.csv', index=False)
        
        # Save as Markdown
        md_table = "# Model Performance Summary\n\n"
        md_table += "| Metric | Value |\n|--------|-------|\n"
        for key, value in summary.items():
            md_table += f"| {key} | {value} |\n"
        
        with open(metrics_dir / 'summary.md', 'w') as f:
            f.write(md_table)
    
    def _create_detailed_report(self, results: Dict, metrics_dir: Path):
        """Create detailed performance report"""
        acc = results.get('accuracy_metrics', {})
        comp = results.get('computational_metrics', {})
        inf = results.get('inference_metrics', {})
        train = results.get('training_metrics', {})
        gen = results.get('generalization_metrics', {})
        
        report = f"""# Detailed Performance Report: {self.model_name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Architecture
- **Parameters**: {comp.get('total_parameters', 0):,}
- **GFLOPs**: {comp.get('gflops', 0):.2f}
- **Memory**: {comp.get('peak_memory_gb', 0):.1f} GB

## Accuracy Metrics (Test Set)
- **R² Score**: {acc.get('r2', 0):.4f}
- **MSE**: {acc.get('mse', 0):.6f}
- **MAE**: {acc.get('mae', 0):.2f} Pa
- **RMSE**: {acc.get('rmse', 0):.2f} Pa
- **Max Error**: {acc.get('max_error', 0):.2f} Pa

## Training Performance
- **Total Time**: {train.get('total_training_time_hours', 0):.2f} hours
- **Best Epoch**: {train.get('best_epoch', 'N/A')}
- **Best Loss**: {train.get('best_loss', 0):.6f}

## Inference Performance
- **Mean Time**: {inf.get('mean_inference_time_ms', 0):.2f} ms
- **Throughput**: {1000/max(inf.get('mean_inference_time_ms', 1), 1):.1f} samples/sec

## Generalization
- **Train-Test MSE Gap**: {gen.get('train_test_mse_gap', 0):.6f}
- **Train-Test R² Gap**: {gen.get('train_test_r2_gap', 0):.4f}

## Performance Assessment
"""
        
        r2 = acc.get('r2', 0)
        if r2 > 0.95:
            report += "- **Excellent**: R² > 0.95 indicates very high accuracy\n"
        elif r2 > 0.90:
            report += "- **Good**: R² > 0.90 indicates good accuracy\n"
        elif r2 > 0.80:
            report += "- **Fair**: R² > 0.80 indicates acceptable accuracy\n"
        else:
            report += "- **Poor**: R² < 0.80 requires improvement\n"
        
        with open(metrics_dir / 'detailed_report.md', 'w') as f:
            f.write(report)


def run_training(args, logger):
    """Run complete training and evaluation pipeline."""
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(project_root, 'models', args.model, 'config.yaml')
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        config = {}
    
    # Override with command line arguments
    config['dataset_path'] = args.dataset_path
    config['subset_dir'] = args.subset_dir
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr
    
    # Initialize components
    logger.info(f"Initializing {args.model}...")
    analyzer = ModelAgnosticAnalyzer(args.save_dir, args.model)
    wrapper = load_model_wrapper(args.model, config, logger)
    evaluator = UnifiedEvaluator(wrapper, save_dir=args.save_dir)
    
    # Setup
    logger.info("Setting up model and data...")
    evaluator.setup()
    
    # Train and evaluate
    logger.info("Starting training...")
    start_time = time.time()
    
    results = evaluator.train_and_evaluate()
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    
    # Save analysis
    analyzer.save_learning_curves(evaluator, args.model)
    analyzer.save_comprehensive_metrics(results, evaluator, args.model)
    
    return results


def run_evaluation(args, logger):
    """Run evaluation only (no training)."""
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(project_root, 'models', args.model, 'config.yaml')
    
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = {}
    
    # Override with command line arguments
    config['dataset_path'] = args.dataset_path
    config['subset_dir'] = args.subset_dir
    
    # Initialize components
    logger.info(f"Initializing {args.model} for evaluation...")
    wrapper = load_model_wrapper(args.model, config, logger)
    evaluator = UnifiedEvaluator(wrapper, save_dir=args.save_dir, enable_bootstrap=args.enable_bootstrap)
    
    # Setup
    evaluator.setup()
    
    # Evaluate
    results = evaluator.evaluate_single_model(checkpoint_path=args.checkpoint)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Model-agnostic training and evaluation framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('model', type=str, 
                       help='Model name (folder name in models/)')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to DrivAerNet++ dataset')
    parser.add_argument('--subset-dir', type=str, required=True,
                       help='Path to train/val/test splits')
    
    # Mode selection
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate model (no training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--enable-bootstrap', action='store_true',
                       help='Enable bootstrap confidence intervals (B=2000)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("MODEL-AGNOSTIC FRAMEWORK")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Mode: {'Evaluation' if args.evaluate_only else 'Training'}")
    print(f"Save directory: {args.save_dir}")
    print("="*60 + "\n")
    
    try:
        if args.evaluate_only:
            results = run_evaluation(args, logger)
        else:
            results = run_training(args, logger)
        
        # Print final summary
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)
        print(f"Results saved to: {args.save_dir}")
        
        # Print key metrics
        if 'accuracy_metrics' in results:
            acc = results['accuracy_metrics']
            print(f"\nFinal Test Performance:")
            print(f"  R² Score: {acc.get('r2', 0):.4f}")
            print(f"  MSE: {acc.get('mse', 0):.6f}")
            print(f"  MAE: {acc.get('mae', 0):.2f} Pa")
            print(f"  RMSE: {acc.get('rmse', 0):.2f} Pa")
            print(f"  Relative L2: {acc.get('rel_l2', 0):.6f}")
            print(f"  Max Error: {acc.get('max_error', 0):.2f} Pa")
        
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()