#!/usr/bin/env python
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Model-Agnostic Training and Evaluation Script with AB-UPT Support

NORMALIZATION:
- All metrics are computed in ORIGINAL scale (Pascals) for interpretability
- Models internally handle normalization as needed for training stability
- AB-UPT uses PER-SAMPLE normalization matching the notebook approach
- The framework ensures consistent normalization/denormalization flow
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
import subprocess

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
        return yaml.safe_load(f)


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


def is_abupt_model(model_name: str) -> bool:
    """Check if the model is AB-UPT (requires special handling)."""
    return 'abupt' in model_name.lower() or 'ab-upt' in model_name.lower()


def run_abupt_evaluation(args, logger) -> Dict:
    """Run AB-UPT evaluation using the standalone script."""
    logger.info("="*80)
    logger.info("AB-UPT EVALUATION (Per-Sample Normalization)")
    logger.info("="*80)
    
    # Locate the AB-UPT evaluation script
    model_dir = os.path.join(project_root, 'models', args.model)
    eval_script = os.path.join(model_dir, 'evaluate_abupt_full.py')
    
    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"AB-UPT evaluation script not found at {eval_script}")
    
    logger.info(f"Using evaluation script: {eval_script}")
    
    # CREATE MODEL-SPECIFIC SAVE DIRECTORY - THIS IS THE KEY FIX
    model_save_dir = os.path.join(args.save_dir, args.model.lower())
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {model_save_dir}")
    
    # Prepare command
    cmd = [
        sys.executable,
        eval_script,
        '--dataset-path', args.dataset_path,
        '--subset-dir', args.subset_dir,
        '--checkpoint', args.checkpoint,
        '--save-dir', model_save_dir,  # PASS THE MODEL-SPECIFIC DIR
        '--num-queries', str(args.num_queries) if hasattr(args, 'num_queries') else '10000',
        '--checkpoint-interval', str(args.checkpoint_interval) if hasattr(args, 'checkpoint_interval') else '50',
        '--device', args.device if hasattr(args, 'device') else 'cuda',
        '--splits'
    ]
    
    # Add splits to evaluate
    if args.splits:
        cmd.extend(args.splits)
    else:
        cmd.extend(['train', 'val', 'test'])
    
    # Add verbose flag if needed
    if args.verbose:
        cmd.append('--verbose')
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute the evaluation script
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Log output
        if result.stdout:
            logger.info("Evaluation output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(line)
        
        if result.stderr:
            logger.warning("Evaluation warnings:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(line)
        
        logger.info("✓ AB-UPT evaluation completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"AB-UPT evaluation failed with return code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        raise
    
    # Load results from the saved checkpoint files
    results = load_abupt_results(
        args.save_dir, 
        args.model, 
        args.splits if args.splits else ['train', 'val', 'test'], 
        logger,
        enable_bootstrap=args.enable_bootstrap  # ADD THIS
    )    
    return results


def load_abupt_results(save_dir: str, model_name: str, splits: List[str], 
                       logger, enable_bootstrap: bool = False) -> Dict:
    """Load AB-UPT evaluation results from checkpoint files with optional bootstrap."""
    logger.info("Loading AB-UPT evaluation results...")
    
    results_dir = Path(save_dir) / model_name.lower()
    results = {
        'accuracy_metrics': {},
        'computational_metrics': {},
        'inference_metrics': {},
        'generalization_metrics': {},
        'split_results': {},
        'bootstrap_metrics': {}  # Add this
    }
    
    # Store test data for bootstrap
    test_df = None
    
    # Load results for each split
    for split in splits:
        checkpoint_file = results_dir / f'abupt_{split}_eval_checkpoint.csv'
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint file not found for {split} split: {checkpoint_file}")
            continue
        
        try:
            df = pd.read_csv(checkpoint_file)
            valid_df = df.dropna()
            
            if len(valid_df) == 0:
                logger.warning(f"No valid results in {split} split")
                continue
            
            # Extract metrics (in RAW SPACE - Pascals)
            split_metrics = {
                'r2': valid_df['r2_score'].mean(),
                'r2_std': valid_df['r2_score'].std(),
                'mse_raw': valid_df['mse_raw'].mean(),
                'mae_raw': valid_df['mae_raw'].mean(),
                'rmse_raw': np.sqrt(valid_df['mse_raw'].mean()),
                'rel_l2': valid_df['l2_error_raw'].mean(),
                'rel_l1': valid_df['l1_error_raw'].mean(),
                'max_error': valid_df['mae_raw'].max(),
                'inference_time_ms': valid_df['inference_times'].mean() * 1000,
                'total_samples': len(df),
                'valid_samples': len(valid_df),
                'failed_samples': len(df) - len(valid_df)
            }
            
            results['split_results'][split] = split_metrics
            
            # Store test data for bootstrap
            if split == 'test':
                test_df = valid_df
            
            logger.info(f"✓ Loaded {split} results: R²={split_metrics['r2']:.4f}, "
                       f"Rel L2={split_metrics['rel_l2']:.4f} ({split_metrics['rel_l2']*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error loading {split} results: {e}")
            continue
    
    # Set test metrics as primary accuracy metrics
    if 'test' in results['split_results']:
        test_metrics = results['split_results']['test']
        results['accuracy_metrics'] = {
            'r2': test_metrics['r2'],
            'r2_std': test_metrics['r2_std'],
            'mse': test_metrics['mse_raw'],
            'mae': test_metrics['mae_raw'],
            'rmse': test_metrics['rmse_raw'],
            'rel_l2': test_metrics['rel_l2'],
            'rel_l1': test_metrics['rel_l1'],
            'max_error': test_metrics['max_error']
        }
        results['inference_metrics'] = {
            'mean_inference_time_ms': test_metrics['inference_time_ms']
        }
        
        # Compute bootstrap if enabled
        if enable_bootstrap and test_df is not None and len(test_df) > 0:
            logger.info("Computing bootstrap confidence intervals (B=2000, 95% CI)...")
            results['bootstrap_metrics'] = compute_abupt_bootstrap(test_df, logger)

    # Compute generalization metrics if both train and test are available
    if 'train' in results['split_results'] and 'test' in results['split_results']:
        train_metrics = results['split_results']['train']
        test_metrics = results['split_results']['test']
        
        results['generalization_metrics'] = {
            'train_test_r2_gap': train_metrics['r2'] - test_metrics['r2'],
            'train_test_mse_gap': test_metrics['mse_raw'] - train_metrics['mse_raw'],
            'train_test_rel_l2_gap': test_metrics['rel_l2'] - train_metrics['rel_l2'],
            'overfitting_detected': (train_metrics['r2'] - test_metrics['r2']) > 0.05
        }
        
        logger.info(f"Generalization: R² gap = {results['generalization_metrics']['train_test_r2_gap']:.4f}")
    
    # Add computational metrics (basic)
    results['computational_metrics'] = {
        'model_name': model_name,
        'evaluation_framework': 'AB-UPT (Per-Sample Normalization)',
        'metric_space': 'Raw (Pascals)'
    }
    
    return results

# Add after line 295 (after load_abupt_results function)

def compute_abupt_bootstrap(test_df: pd.DataFrame, logger, n_bootstrap: int = 2000, 
                            confidence: float = 0.95) -> Dict:
    """
    Compute bootstrap confidence intervals for AB-UPT metrics.
    
    Args:
        test_df: DataFrame with per-sample metrics
        logger: Logger instance
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Dictionary with bootstrap statistics
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    n_samples = len(test_df)
    alpha = (1 - confidence) / 2
    
    logger.info(f"Running {n_bootstrap} bootstrap iterations on {n_samples} samples...")
    
    # Metrics to bootstrap
    metrics_to_bootstrap = [
        'r2_score', 'mse_raw', 'mae_raw', 'l2_error_raw', 'l1_error_raw'
    ]
    
    bootstrap_results = {metric: [] for metric in metrics_to_bootstrap}
    
    # Bootstrap resampling
    for i in range(n_bootstrap):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Progress: {i}/{n_bootstrap} iterations...")
        
        # Resample with replacement
        bootstrap_sample = test_df.sample(n=n_samples, replace=True)
        
        # Compute metrics on bootstrap sample
        for metric in metrics_to_bootstrap:
            if metric in bootstrap_sample.columns:
                bootstrap_results[metric].append(bootstrap_sample[metric].mean())
    
    # Compute confidence intervals
    bootstrap_metrics = {}
    
    for metric in metrics_to_bootstrap:
        if len(bootstrap_results[metric]) > 0:
            values = np.array(bootstrap_results[metric])
            
            bootstrap_metrics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, alpha * 100)),
                'ci_upper': float(np.percentile(values, (1 - alpha) * 100)),
                'median': float(np.median(values))
            }
            
            # Log result
            logger.info(f"  {metric}: {bootstrap_metrics[metric]['mean']:.6f} "
                       f"[{bootstrap_metrics[metric]['ci_lower']:.6f}, "
                       f"{bootstrap_metrics[metric]['ci_upper']:.6f}]")
    
    # Compute derived metrics
    if 'mse_raw' in bootstrap_metrics:
        rmse_values = np.sqrt(np.array(bootstrap_results['mse_raw']))
        bootstrap_metrics['rmse_raw'] = {
            'mean': float(np.mean(rmse_values)),
            'std': float(np.std(rmse_values)),
            'ci_lower': float(np.percentile(rmse_values, alpha * 100)),
            'ci_upper': float(np.percentile(rmse_values, (1 - alpha) * 100)),
            'median': float(np.median(rmse_values))
        }
        logger.info(f"  rmse_raw: {bootstrap_metrics['rmse_raw']['mean']:.6f} "
                   f"[{bootstrap_metrics['rmse_raw']['ci_lower']:.6f}, "
                   f"{bootstrap_metrics['rmse_raw']['ci_upper']:.6f}]")
    
    logger.info("✓ Bootstrap confidence intervals computed")
    
    return bootstrap_metrics


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
        with open(curves_dir / 'training_history.json', 'w') as f:
            json.dump(data, f, indent=2)
    
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
        
        # Save split-specific results if available (for AB-UPT)
        if 'split_results' in results and results['split_results']:
            self._save_split_results(results['split_results'], metrics_dir)
        
        self.logger.info(f"Metrics saved to {metrics_dir}")
    
def _save_split_results(self, split_results: Dict, metrics_dir: Path):
    """Save split-specific results for AB-UPT."""
    self.logger.info("Saving split-specific results...")
    
    split_dir = metrics_dir / 'splits'
    split_dir.mkdir(exist_ok=True)
    
    # Create summary table
    summary_data = []
    for split_name, metrics in split_results.items():
        summary_data.append({
            'Split': split_name.upper(),
            'R²': f"{metrics['r2']:.4f}",
            'R² Std': f"{metrics.get('r2_std', 0):.4f}",
            'Rel L2': f"{metrics['rel_l2']:.4f} ({metrics['rel_l2']*100:.2f}%)",
            'Rel L1': f"{metrics['rel_l1']:.4f} ({metrics['rel_l1']*100:.2f}%)",
            'MAE (Pa)': f"{metrics['mae_raw']:.2f}",
            'RMSE (Pa)': f"{metrics['rmse_raw']:.2f}",
            'Samples': f"{metrics['valid_samples']}/{metrics['total_samples']}"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(split_dir / 'split_comparison.csv', index=False)
    
    # Save as markdown (simple format without tabulate)
    with open(split_dir / 'split_comparison.md', 'w') as f:
        f.write("# Split-Specific Results\n\n")
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
    
    self.logger.info(f"✓ Split results saved to {split_dir}/")
    
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
            'Rel L2': f"{acc.get('rel_l2', 0):.4f}",
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

## Accuracy Metrics (Test Set - Physical Units)
- **R² Score**: {acc.get('r2', 0):.4f}
- **MSE**: {acc.get('mse', 0):.6f} Pa²
- **MAE**: {acc.get('mae', 0):.2f} Pa
- **RMSE**: {acc.get('rmse', 0):.2f} Pa
- **Relative L2**: {acc.get('rel_l2', 0):.4f} ({acc.get('rel_l2', 0)*100:.2f}%)
- **Relative L1**: {acc.get('rel_l1', 0):.4f} ({acc.get('rel_l1', 0)*100:.2f}%)
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
- **Overfitting Detected**: {'Yes' if gen.get('overfitting_detected', False) else 'No'}

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
        
        # Add AB-UPT specific note if applicable
        if 'evaluation_framework' in comp and 'Per-Sample' in comp['evaluation_framework']:
            report += "\n## Evaluation Framework\n"
            report += "- **Framework**: " + comp['evaluation_framework'] + "\n"
            report += "- **Metric Space**: " + comp.get('metric_space', 'N/A') + "\n"
            report += "- **Note**: Uses per-sample normalization matching notebook evaluation\n"
        
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
    # Check if this is an AB-UPT model
    if is_abupt_model(args.model):
        logger.info("Detected AB-UPT model - using specialized evaluation")
        return run_abupt_evaluation(args, logger)
    
    # Standard evaluation for other models
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
        description='Model-agnostic training and evaluation framework with AB-UPT support',
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
    
    # AB-UPT specific arguments
    parser.add_argument('--num-queries', type=int, default=10000,
                       help='Number of query points for AB-UPT evaluation')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Checkpoint interval for AB-UPT evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                       help='Splits to evaluate (train, val, test)')
    
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
    print("\n" + "="*80)
    print("MODEL-AGNOSTIC FRAMEWORK WITH AB-UPT SUPPORT")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Mode: {'Evaluation' if args.evaluate_only else 'Training'}")
    print(f"Save directory: {args.save_dir}")
    if is_abupt_model(args.model):
        print(f"⚠️  AB-UPT detected: Using per-sample normalization")
    print("="*80 + "\n")
    
    try:
        if args.evaluate_only:
            results = run_evaluation(args, logger)
            
            # Create analyzer for saving results
            analyzer = ModelAgnosticAnalyzer(args.save_dir, args.model)
            analyzer.save_comprehensive_metrics(results, None, args.model)
        else:
            results = run_training(args, logger)
        
        # Print final summary
        print("\n" + "="*80)
        print("EXECUTION COMPLETE")
        print("="*80)
        print(f"Results saved to: {args.save_dir}")
        
        # Print key metrics
        if 'accuracy_metrics' in results:
            acc = results['accuracy_metrics']
            print(f"\nFinal Test Performance:")
            print(f"  R² Score: {acc.get('r2', 0):.4f}")
            print(f"  MSE: {acc.get('mse', 0):.6f} Pa²")
            print(f"  MAE: {acc.get('mae', 0):.2f} Pa")
            print(f"  RMSE: {acc.get('rmse', 0):.2f} Pa")
            print(f"  Relative L2: {acc.get('rel_l2', 0):.4f} ({acc.get('rel_l2', 0)*100:.2f}%)")
            print(f"  Max Error: {acc.get('max_error', 0):.2f} Pa")
        
        # Print split-specific results if available
        if 'split_results' in results and results['split_results']:
            print(f"\nSplit-Specific Results:")
            for split_name, split_metrics in results['split_results'].items():
                print(f"\n  {split_name.upper()}:")
                print(f"    R²: {split_metrics['r2']:.4f}")
                print(f"    Rel L2: {split_metrics['rel_l2']:.4f} ({split_metrics['rel_l2']*100:.2f}%)")
                print(f"    Samples: {split_metrics['valid_samples']}/{split_metrics['total_samples']}")
        
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()