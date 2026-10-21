#!/usr/bin/env python3
"""
Cross-Category Generalization Experiments for DrivAerNet++
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu

This script tests the generalization ability of Transolver and Transolver++ 
across different car body types (Estateback, Notchback, Fastback).

Experiments:
1. Train on E, test on N and F
2. Train on N, test on E and F  
3. Train on F, test on E and N
4. Train on E+N, test on F
5. Train on E+F, test on N
6. Train on N+F, test on E
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.cross_category_evaluator import CrossCategoryEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarCategoryClassifier:
    """Classify car designs by body type from filename."""
    
    CATEGORIES = {
        'E': 'Estateback',
        'N': 'Notchback', 
        'F': 'Fastback'
    }
    
    @staticmethod
    def get_category(design_id: str) -> str:
        """
        Extract category from design ID.
        
        Format: CATEGORY_SIDE_WHEELS_MIRRORS_ID
        Example: E_S_WW_WM_648 -> E (Estateback)
        
        Args:
            design_id: Design identifier string
            
        Returns:
            Category code ('E', 'N', or 'F')
        """
        parts = design_id.split('_')
        if len(parts) > 0 and parts[0] in CarCategoryClassifier.CATEGORIES:
            return parts[0]
        else:
            logger.warning(f"Could not extract category from {design_id}, defaulting to 'E'")
            return 'E'
    
    @staticmethod
    def get_category_name(category_code: str) -> str:
        """Get full category name from code."""
        return CarCategoryClassifier.CATEGORIES.get(category_code, 'Unknown')
    
    @staticmethod
    def split_by_category(design_ids: List[str]) -> Dict[str, List[str]]:
        """Split design IDs by category."""
        splits = {'E': [], 'N': [], 'F': []}
        
        for design_id in design_ids:
            category = CarCategoryClassifier.get_category(design_id)
            if category in splits:
                splits[category].append(design_id)
        
        return splits


def load_design_ids(file_path: str) -> List[str]:
    """Load design IDs from file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def create_category_splits(
    dataset_path: str,
    subset_dir: str,
    output_dir: str
) -> Dict[str, Dict[str, str]]:
    """
    Create train/val/test splits for each cross-category experiment.
    
    Returns:
        Dictionary mapping experiment name to split file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original splits
    train_ids = load_design_ids(Path(subset_dir) / 'train_design_ids.txt')
    val_ids = load_design_ids(Path(subset_dir) / 'val_design_ids.txt')
    test_ids = load_design_ids(Path(subset_dir) / 'test_design_ids.txt')
    
    # Combine all for category-based splitting
    all_ids = train_ids + val_ids + test_ids
    
    # Split by category
    category_splits = CarCategoryClassifier.split_by_category(all_ids)
    
    logger.info("Category distribution:")
    for cat, ids in category_splits.items():
        cat_name = CarCategoryClassifier.get_category_name(cat)
        logger.info(f"  {cat_name} ({cat}): {len(ids)} samples")
    
    experiments = {}
    
    # Single category training experiments
    for train_cat in ['E', 'N', 'F']:
        test_cats = [c for c in ['E', 'N', 'F'] if c != train_cat]
        
        exp_name = f"train_{train_cat}_test_{''.join(test_cats)}"
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Train/val from single category (80/20 split)
        train_cat_ids = category_splits[train_cat]
        np.random.shuffle(train_cat_ids)
        
        n_train = int(0.8 * len(train_cat_ids))
        train_split = train_cat_ids[:n_train]
        val_split = train_cat_ids[n_train:]
        
        # Test from other categories
        test_split = []
        for test_cat in test_cats:
            test_split.extend(category_splits[test_cat])
        
        # Save splits
        train_file = exp_dir / 'train_design_ids.txt'
        val_file = exp_dir / 'val_design_ids.txt'
        test_file = exp_dir / 'test_design_ids.txt'
        
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_split))
        with open(val_file, 'w') as f:
            f.write('\n'.join(val_split))
        with open(test_file, 'w') as f:
            f.write('\n'.join(test_split))
        
        experiments[exp_name] = {
            'train_file': str(train_file),
            'val_file': str(val_file),
            'test_file': str(test_file),
            'train_cats': [train_cat],
            'test_cats': test_cats,
            'n_train': len(train_split),
            'n_val': len(val_split),
            'n_test': len(test_split)
        }
        
        logger.info(f"Created {exp_name}: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}")
    
    # Two category training experiments
    for test_cat in ['E', 'N', 'F']:
        train_cats = [c for c in ['E', 'N', 'F'] if c != test_cat]
        
        exp_name = f"train_{''.join(train_cats)}_test_{test_cat}"
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Train/val from two categories (combined, then 80/20 split)
        train_cat_ids = []
        for train_cat in train_cats:
            train_cat_ids.extend(category_splits[train_cat])
        
        np.random.shuffle(train_cat_ids)
        
        n_train = int(0.8 * len(train_cat_ids))
        train_split = train_cat_ids[:n_train]
        val_split = train_cat_ids[n_train:]
        
        # Test from remaining category
        test_split = category_splits[test_cat]
        
        # Save splits
        train_file = exp_dir / 'train_design_ids.txt'
        val_file = exp_dir / 'val_design_ids.txt'
        test_file = exp_dir / 'test_design_ids.txt'
        
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_split))
        with open(val_file, 'w') as f:
            f.write('\n'.join(val_split))
        with open(test_file, 'w') as f:
            f.write('\n'.join(test_split))
        
        experiments[exp_name] = {
            'train_file': str(train_file),
            'val_file': str(val_file),
            'test_file': str(test_file),
            'train_cats': train_cats,
            'test_cats': [test_cat],
            'n_train': len(train_split),
            'n_val': len(val_split),
            'n_test': len(test_split)
        }
        
        logger.info(f"Created {exp_name}: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}")
    
    # Save experiment metadata
    metadata_file = output_dir / 'experiments_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    logger.info(f"Created {len(experiments)} cross-category experiments")
    logger.info(f"Metadata saved to {metadata_file}")
    
    return experiments


def run_experiment(
    model_name: str,
    experiment_name: str,
    experiment_config: Dict,
    dataset_path: str,
    save_dir: str,
    config_path: str,
    batch_size: int = 8,
    epochs: int = 200,
    cuda_devices: str = "0,1,2,3",
    skip_completed: bool = True
) -> Dict:
    """Run a single cross-category experiment."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {model_name} - {experiment_name}")
    logger.info(f"Train categories: {experiment_config['train_cats']}")
    logger.info(f"Test categories: {experiment_config['test_cats']}")
    logger.info(f"{'='*80}\n")
    
    # Create experiment-specific save directory
    exp_save_dir = Path(save_dir) / model_name / experiment_name
    exp_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if experiment is already completed
    # Look for comprehensive_benchmark_summary.csv which is created after successful training
    results_csv = exp_save_dir / 'comprehensive_benchmark_summary.csv'
    
    if skip_completed and results_csv.exists():
        logger.info(f"⏭️  Experiment already completed, loading existing results from {results_csv}")
        try:
            # Load results from CSV
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            if len(df) > 0:
                row = df.iloc[0]
                results = {
                    'mae': float(row['test_mae']),
                    'rmse': float(row['test_rmse']),
                    'r2': float(row['test_r2']),
                    'relative_l2': float(row['test_rel_l2']),
                    'mse': float(row['test_mse']),
                    'max_error': float(row['test_max_error']),
                    'experiment': experiment_name,
                    'model': model_name,
                    'train_categories': experiment_config['train_cats'],
                    'test_categories': experiment_config['test_cats']
                }
                
                # Print results
                print_experiment_results(results)
                
                return results
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}. Re-running experiment...")
    
    # Create experiment-specific subset directory (not temporary)
    exp_subset_dir = exp_save_dir / 'splits'
    exp_subset_dir.mkdir(exist_ok=True)
    
    # Copy split files to experiment directory
    import shutil
    shutil.copy2(experiment_config['train_file'], exp_subset_dir / 'train_design_ids.txt')
    shutil.copy2(experiment_config['val_file'], exp_subset_dir / 'val_design_ids.txt')
    shutil.copy2(experiment_config['test_file'], exp_subset_dir / 'test_design_ids.txt')
    
    # Run training using train_and_analyze.py
    cmd = [
        'python', 'scripts/train_and_analyze.py',
        model_name,
        '--dataset-path', dataset_path,
        '--subset-dir', str(exp_subset_dir),
        '--save-dir', str(exp_save_dir),
        '--config', config_path,
        '--batch-size', str(batch_size),
        '--epochs', str(epochs)
    ]
    
    # Set CUDA devices
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices
    
    import subprocess
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Starting training (output will be shown in real-time)...\n")
    sys.stdout.flush()
    
    # Run with real-time output (no capture)
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        logger.error(f"\n❌ Training failed for {experiment_name} with exit code {result.returncode}")
        return None
    
    logger.info(f"\n✅ Training completed successfully for {experiment_name}")
    
    # Load results from comprehensive_benchmark_summary.csv
    results_csv = exp_save_dir / 'comprehensive_benchmark_summary.csv'
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            if len(df) > 0:
                row = df.iloc[0]
                results = {
                    'mae': float(row['test_mae']),
                    'rmse': float(row['test_rmse']),
                    'r2': float(row['test_r2']),
                    'relative_l2': float(row['test_rel_l2']),
                    'mse': float(row['test_mse']),
                    'max_error': float(row['test_max_error']),
                    'experiment': experiment_name,
                    'model': model_name,
                    'train_categories': experiment_config['train_cats'],
                    'test_categories': experiment_config['test_cats']
                }
                
                # Print results immediately
                print_experiment_results(results)
                
                return results
        except Exception as e:
            logger.error(f"Failed to load results from CSV: {e}")
    
    logger.error(f"Results file not found: {results_csv}")
    return None


def print_experiment_results(results: Dict):
    """Print experiment results in a formatted way."""
    print("\n" + "="*80)
    print(f"📊 RESULTS: {results['model']} - {results['experiment']}")
    print("="*80)
    
    train_cats = ''.join(results['train_categories'])
    test_cats = ''.join(results['test_categories'])
    print(f"Train Categories: {train_cats} ({', '.join([CarCategoryClassifier.get_category_name(c) for c in results['train_categories']])})")
    print(f"Test Categories:  {test_cats} ({', '.join([CarCategoryClassifier.get_category_name(c) for c in results['test_categories']])})")
    print("-"*80)
    
    print(f"MAE:         {results.get('mae', 'N/A'):.4f}")
    print(f"RMSE:        {results.get('rmse', 'N/A'):.4f}")
    print(f"R²:          {results.get('r2', 'N/A'):.6f}")
    print(f"Relative L2: {results.get('relative_l2', 'N/A'):.6f}")
    
    if 'mape' in results:
        print(f"MAPE:        {results['mape']:.4f}%")
    if 'max_error' in results:
        print(f"Max Error:   {results['max_error']:.4f}")
    
    print("="*80 + "\n")
    sys.stdout.flush()


def print_model_summary(model_name: str, results: List[Dict]):
    """Print summary of all experiments for a model."""
    print("\n" + "🌟"*40)
    print(f"{'':^80}")
    print(f"{f'SUMMARY: {model_name.upper()}':^80}")
    print(f"{'':^80}")
    print("🌟"*40)
    
    # Create summary table
    print(f"\n{'Experiment':<25} {'MAE':>10} {'RMSE':>10} {'R²':>12} {'Rel L2':>12}")
    print("-"*80)
    
    for result in sorted(results, key=lambda x: x['experiment']):
        exp_name = result['experiment'].replace('train_', '').replace('_test_', '→')
        mae = result.get('mae', float('nan'))
        rmse = result.get('rmse', float('nan'))
        r2 = result.get('r2', float('nan'))
        rel_l2 = result.get('relative_l2', float('nan'))
        
        print(f"{exp_name:<25} {mae:>10.4f} {rmse:>10.4f} {r2:>12.6f} {rel_l2:>12.6f}")
    
    # Calculate averages
    avg_mae = np.mean([r.get('mae', float('nan')) for r in results])
    avg_rmse = np.mean([r.get('rmse', float('nan')) for r in results])
    avg_r2 = np.mean([r.get('r2', float('nan')) for r in results])
    avg_rel_l2 = np.mean([r.get('relative_l2', float('nan')) for r in results])
    
    print("-"*80)
    print(f"{'AVERAGE':<25} {avg_mae:>10.4f} {avg_rmse:>10.4f} {avg_r2:>12.6f} {avg_rel_l2:>12.6f}")
    print("="*80 + "\n")
    sys.stdout.flush()


def print_final_comparison(all_results: List[Dict]):
    """Print final comparison between all models."""
    print("\n" + "🏆"*40)
    print(f"{'':^80}")
    print(f"{'FINAL COMPARISON - ALL MODELS':^80}")
    print(f"{'':^80}")
    print("🏆"*40)
    
    models = sorted(set(r['model'] for r in all_results))
    
    print(f"\n{'Model':<20} {'Avg MAE':>12} {'Avg RMSE':>12} {'Avg R²':>12} {'Avg Rel L2':>12}")
    print("-"*80)
    
    for model in models:
        model_results = [r for r in all_results if r['model'] == model]
        
        avg_mae = np.mean([r.get('mae', float('nan')) for r in model_results])
        avg_rmse = np.mean([r.get('rmse', float('nan')) for r in model_results])
        avg_r2 = np.mean([r.get('r2', float('nan')) for r in model_results])
        avg_rel_l2 = np.mean([r.get('relative_l2', float('nan')) for r in model_results])
        
        print(f"{model:<20} {avg_mae:>12.4f} {avg_rmse:>12.4f} {avg_r2:>12.6f} {avg_rel_l2:>12.6f}")
    
    # Find best model for each metric
    print("\n" + "-"*80)
    print("🥇 Best Performance:")
    print("-"*80)
    
    model_avgs = {}
    for model in models:
        model_results = [r for r in all_results if r['model'] == model]
        model_avgs[model] = {
            'mae': np.mean([r.get('mae', float('nan')) for r in model_results]),
            'rmse': np.mean([r.get('rmse', float('nan')) for r in model_results]),
            'r2': np.mean([r.get('r2', float('nan')) for r in model_results]),
            'relative_l2': np.mean([r.get('relative_l2', float('nan')) for r in model_results])
        }
    
    best_mae = min(model_avgs.items(), key=lambda x: x[1]['mae'])
    best_rmse = min(model_avgs.items(), key=lambda x: x[1]['rmse'])
    best_r2 = max(model_avgs.items(), key=lambda x: x[1]['r2'])
    best_rel_l2 = min(model_avgs.items(), key=lambda x: x[1]['relative_l2'])
    
    print(f"  MAE:         {best_mae[0]:<20} ({best_mae[1]['mae']:.4f})")
    print(f"  RMSE:        {best_rmse[0]:<20} ({best_rmse[1]['rmse']:.4f})")
    print(f"  R²:          {best_r2[0]:<20} ({best_r2[1]['r2']:.6f})")
    print(f"  Relative L2: {best_rel_l2[0]:<20} ({best_rel_l2[1]['relative_l2']:.6f})")
    
    print("="*80 + "\n")
    sys.stdout.flush()




def visualize_results(
    all_results: List[Dict],
    output_dir: str
):
    """Create visualizations of cross-category generalization results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save raw results
    df.to_csv(output_dir / 'cross_category_results.csv', index=False)
    
    # Metrics to plot
    metrics = ['mae', 'rmse', 'r2', 'relative_l2']
    metric_names = {
        'mae': 'MAE (m²/s²)',
        'rmse': 'RMSE (m²/s²)',
        'r2': 'R²',
        'relative_l2': 'Relative L2'
    }
    
    # 1. Bar plot comparing models across experiments
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for plotting
        exp_order = sorted(df['experiment'].unique())
        models = sorted(df['model'].unique())
        
        x = np.arange(len(exp_order))
        width = 0.35
        
        for i, model in enumerate(models):
            model_data = df[df['model'] == model]
            values = [model_data[model_data['experiment'] == exp][metric].values[0] 
                     if len(model_data[model_data['experiment'] == exp]) > 0 else 0
                     for exp in exp_order]
            
            ax.bar(x + i*width, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.set_title(f'{metric_names[metric]} across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(exp_order, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot to {output_dir / 'cross_category_comparison.png'}")
    
    # 2. Heatmap of generalization performance
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, model in enumerate(sorted(df['model'].unique())):
            model_df = df[df['model'] == model]
            
            # Create matrix: rows=train config, cols=test category
            train_configs = ['E', 'N', 'F', 'EN', 'EF', 'NF']
            test_cats = ['E', 'N', 'F']
            
            matrix = np.zeros((len(train_configs), len(test_cats)))
            
            for i, train_config in enumerate(train_configs):
                for j, test_cat in enumerate(test_cats):
                    # Find matching experiment
                    if len(train_config) == 1:
                        exp_name = f"train_{train_config}_test_{''.join([c for c in test_cats if c != train_config])}"
                        # Check if test_cat is in the test set
                        if test_cat != train_config:
                            matching = model_df[model_df['experiment'] == exp_name]
                            if len(matching) > 0:
                                matrix[i, j] = matching[metric].values[0]
                    else:
                        # Two category training
                        remaining_cat = [c for c in test_cats if c not in train_config][0]
                        exp_name = f"train_{train_config}_test_{remaining_cat}"
                        if test_cat == remaining_cat:
                            matching = model_df[model_df['experiment'] == exp_name]
                            if len(matching) > 0:
                                matrix[i, j] = matching[metric].values[0]
            
            ax = axes[idx]
            im = ax.imshow(matrix, cmap='RdYlGn_r' if metric != 'r2' else 'RdYlGn', aspect='auto')
            
            ax.set_xticks(np.arange(len(test_cats)))
            ax.set_yticks(np.arange(len(train_configs)))
            ax.set_xticklabels(test_cats)
            ax.set_yticklabels(train_configs)
            
            ax.set_xlabel('Test Category', fontsize=12)
            ax.set_ylabel('Train Configuration', fontsize=12)
            ax.set_title(f'{model} - {metric_names[metric]}', fontsize=14, fontweight='bold')
            
            # Add values in cells
            for i in range(len(train_configs)):
                for j in range(len(test_cats)):
                    if matrix[i, j] != 0:
                        text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=10)
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'generalization_heatmap_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap for {metric}")
    
    # 3. Summary statistics table
    summary = df.groupby(['model', 'experiment'])[metrics].agg(['mean', 'std'])
    summary.to_csv(output_dir / 'cross_category_summary.csv')
    
    logger.info(f"Saved summary statistics to {output_dir / 'cross_category_summary.csv'}")
    
    # 4. Create LaTeX table
    latex_table = create_latex_table(df, metrics)
    with open(output_dir / 'cross_category_table.tex', 'w') as f:
        f.write(latex_table)
    
    logger.info(f"Saved LaTeX table to {output_dir / 'cross_category_table.tex'}")


def create_latex_table(df: pd.DataFrame, metrics: List[str]) -> str:
    """Create LaTeX table for publication."""
    
    latex = r"""\begin{table*}[t]
\centering
\caption{Cross-Category Generalization Performance on DrivAerNet++}
\label{tab:cross_category}
\small
\begin{tabular}{ll|cccc}
\hline
\textbf{Model} & \textbf{Experiment} & \textbf{MAE} & \textbf{RMSE} & \textbf{R²} & \textbf{Rel L2} \\
\hline
"""
    
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        latex += f"\\multirow{{{len(model_df)}}}{{*}}{{{model}}} "
        
        for idx, row in model_df.iterrows():
            exp = row['experiment']
            train_cats = ''.join(row['train_categories'])
            test_cats = ''.join(row['test_categories'])
            
            if idx != model_df.index[0]:
                latex += "& "
            
            latex += f"& {train_cats}$\\to${test_cats} "
            latex += f"& {row['mae']:.2f} "
            latex += f"& {row['rmse']:.2f} "
            latex += f"& {row['r2']:.4f} "
            latex += f"& {row['relative_l2']:.4f} \\\\\n"
        
        latex += "\\hline\n"
    
    latex += r"""\end{tabular}
\vspace{0.5em}
\begin{minipage}{\textwidth}
\small
\textbf{Notes:} 
E=Estateback, N=Notchback, F=Fastback. 
Format: Train$\to$Test indicates training categories and test categories.
Single category experiments test generalization to unseen body types.
Two category experiments test generalization to the held-out type.
\end{minipage}
\end{table*}
"""
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Cross-Category Generalization Experiments')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to DrivAerNet dataset')
    parser.add_argument('--subset-dir', type=str, required=True,
                       help='Path to original train/val/test splits')
    parser.add_argument('--save-dir', type=str, default='results/cross_category',
                       help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--cuda-devices', type=str, default='0,1,2,3',
                       help='CUDA devices to use')
    parser.add_argument('--models', nargs='+', default=['transolver', 'transolver_plus'],
                       help='Models to evaluate')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiments to run (default: all)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only generate visualizations')
    parser.add_argument('--skip-completed', action='store_true', default=True,
                       help='Skip already completed experiments (default: True)')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force re-run all experiments even if already completed')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Cross-Category Generalization Experiments")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info("="*80)
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category splits
    splits_dir = save_dir / 'category_splits'
    logger.info("\nCreating category-based train/val/test splits...")
    experiments = create_category_splits(
        dataset_path=args.dataset_path,
        subset_dir=args.subset_dir,
        output_dir=str(splits_dir)
    )
    
    # Filter experiments if specified
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}
        logger.info(f"Running {len(experiments)} specified experiments")
    
    # Run experiments
    all_results = []
    
    if not args.skip_training:
        total_experiments = len(args.models) * len(experiments)
        current_exp = 0
        skip_completed = not args.force_rerun  # Skip completed unless force_rerun is True
        
        if skip_completed:
            logger.info("✅ Skip-completed mode: Will resume from last completed experiment")
        else:
            logger.info("🔄 Force-rerun mode: Will re-run all experiments from scratch")
        
        for model_name in args.models:
            # Get model config path
            config_path = f"models/{model_name}/config.yaml"
            
            if not os.path.exists(config_path):
                logger.error(f"Config not found: {config_path}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Running experiments for {model_name}")
            logger.info(f"{'='*80}\n")
            
            for exp_idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
                current_exp += 1
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"🔬 EXPERIMENT {current_exp}/{total_experiments}")
                logger.info(f"Model: {model_name} | Experiment: {exp_name}")
                logger.info(f"Train: {exp_config['train_cats']} | Test: {exp_config['test_cats']}")
                logger.info(f"{'#'*80}\n")
                
                result = run_experiment(
                    model_name=model_name,
                    experiment_name=exp_name,
                    experiment_config=exp_config,
                    dataset_path=args.dataset_path,
                    save_dir=str(save_dir),
                    config_path=config_path,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    cuda_devices=args.cuda_devices,
                    skip_completed=skip_completed
                )
                
                if result:
                    all_results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_file = save_dir / 'all_results.json'
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                    logger.info(f"✅ Progress: {current_exp}/{total_experiments} experiments completed")
                    logger.info(f"📁 Intermediate results saved to {results_file}\n")
                else:
                    logger.warning(f"⚠️ Experiment failed, continuing to next...")
                
                sys.stdout.flush()
            
            # Print summary for this model
            model_results = [r for r in all_results if r['model'] == model_name]
            if model_results:
                print_model_summary(model_name, model_results)
        
        # Save all results
        results_file = save_dir / 'all_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\nSaved all results to {results_file}")
    else:
        # Load existing results
        results_file = save_dir / 'all_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            logger.info(f"Loaded {len(all_results)} existing results")
        else:
            logger.error("No existing results found. Remove --skip-training flag.")
            return
    
    # Visualize results
    if all_results:
        logger.info("\nGenerating visualizations...")
        viz_dir = save_dir / 'visualizations'
        visualize_results(all_results, str(viz_dir))
        
        # Print final comparison
        print_final_comparison(all_results)
        
        logger.info("\n" + "="*80)
        logger.info("Cross-Category Generalization Experiments Complete!")
        logger.info(f"Results saved to: {save_dir}")
        logger.info(f"Visualizations saved to: {viz_dir}")
        logger.info("="*80)
    else:
        logger.error("No results to visualize")


if __name__ == '__main__':
    main()
