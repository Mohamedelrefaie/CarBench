#!/usr/bin/env python3
"""
Script to plot training and validation loss curves for different ML models.
Loss curves are read from results/model_name/learning_curves/training_curves.csv
and all plots are saved to a single output folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


# Hardcoded model names - add or remove models as needed
# Models in the published figure. PointNet, TripNet and AB-UPT are deliberately
# excluded: PointNet was dropped from the figure, and TripNet/AB-UPT are trained by
# their own reference implementations rather than the common harness, so their loss
# scales are not comparable with these. Pass --models to plot a different set, e.g.
#   --models pointnet neuraloperator ... tripnet abupt
MODEL_NAMES = [
    'neuraloperator',
    'pointmae',
    'pointnetlarge',
    'regdgcnn',
    'point_transformer',
    'transolver_plus',
    'transolver',
    'transolver_large']

# Color scheme for consistent plotting
COLORS = {
    'train': '#2E86AB',  # Blue
    'val': '#A23B72',    # Purple
}

# Model display names (for prettier labels)
MODEL_DISPLAY_NAMES = {
    'pointnet': 'PointNet',
    'pointnetlarge': 'PointNet Large',
    'pointmae': 'PointMAE',
    'regdgcnn': 'RegDGCNN',
    'neuraloperator': 'NeuralOperator',
    'transolver': 'Transolver',
    'transolver_plus': 'Transolver++',
    'point_transformer': 'Point Transformer',
    'transolver_large': 'Transolver Large',
    'tripnet': 'TripNet',
    'abupt': 'AB-UPT',
}


def apply_axis_scale(ax, scale):
    """Apply axis scaling to a loss-curve axis.

    scale: 'linear', 'semilogy' (log loss only), or 'loglog' (log epoch + log loss).
    Epochs start at 1 and losses are strictly positive, so log axes are well-defined.
    On log axes we also enable minor gridlines for readability.
    """
    if scale == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif scale == 'semilogy':
        ax.set_yscale('log')
    elif scale != 'linear':
        raise ValueError(f"Unknown scale '{scale}' (use linear|semilogy|loglog)")
    if scale in ('loglog', 'semilogy'):
        ax.grid(True, which='both', alpha=0.3)


def load_training_curves(model_name, results_dir='results'):
    """
    Load training curves CSV file for a given model.
    
    Args:
        model_name: Name of the model
        results_dir: Base results directory
        
    Returns:
        DataFrame with training curves or None if not found
    """
    csv_path = Path(results_dir) / model_name / 'learning_curves' / 'training_curves.csv'
    
    if not csv_path.exists():
        print(f"⚠️  Warning: Training curves not found for {model_name} at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} epochs for {model_name}")
        return df
    except Exception as e:
        print(f"⚠️  Error loading {csv_path}: {e}")
        return None


def plot_individual_model(df, model_name, output_dir, scale='loglog'):
    """
    Plot training and validation loss curves for a single model.

    Args:
        df: DataFrame with columns 'epoch', 'train_loss', 'val_loss'
        model_name: Name of the model
        output_dir: Directory to save the plot
        scale: axis scaling ('linear', 'semilogy', or 'loglog')
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and validation loss
    ax.plot(df['epoch'], df['train_loss'], 
            color=COLORS['train'], linewidth=2, label='Training Loss')
    ax.plot(df['epoch'], df['val_loss'], 
            color=COLORS['val'], linewidth=2, label='Validation Loss')
    
    # Find best epoch (minimum validation loss), when a validation curve exists
    has_val = df['val_loss'].notna().any()
    if has_val:
        best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
        best_val_loss = df['val_loss'].min()
        ax.axvline(x=best_epoch, color='gray', linestyle='--',
                   linewidth=1, alpha=0.5, label=f'Best Epoch: {int(best_epoch)}')
    
    # Formatting
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{display_name} - Training Progress', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    apply_axis_scale(ax, scale)

    # Add text box with best epoch info
    textstr = (f'Best Val Loss: {best_val_loss:.6f}\nBest Epoch: {int(best_epoch)}'
               if has_val else 'No validation loss logged')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'{model_name}_loss_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot: {output_path}")


def plot_all_models_comparison(model_data, output_dir, scale='loglog'):
    """
    Plot training and validation loss curves for all models on the same plot.

    Args:
        model_data: Dictionary mapping model names to DataFrames
        output_dir: Directory to save the plots
        scale: axis scaling ('linear', 'semilogy', or 'loglog')
    """
    if not model_data:
        print("⚠️  No model data to plot")
        return
    
    # Create color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
    
    # Plot 1: Training Loss Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, (model_name, df) in enumerate(model_data.items()):
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(df['epoch'], df['train_loss'], 
                color=colors[idx], linewidth=2, label=display_name, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    apply_axis_scale(ax, scale)

    plt.tight_layout()
    output_path = Path(output_dir) / 'all_models_train_loss_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {output_path}")
    
    # Plot 2: Validation Loss Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, (model_name, df) in enumerate(model_data.items()):
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(df['epoch'], df['val_loss'], 
                color=colors[idx], linewidth=2, label=display_name, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    apply_axis_scale(ax, scale)

    plt.tight_layout()
    output_path = Path(output_dir) / 'all_models_val_loss_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {output_path}")
    
    # Plot 3: Combined view (Train + Val) for each model
    n_models = len(model_data)
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, df) in enumerate(model_data.items()):
        ax = axes[idx]
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        ax.plot(df['epoch'], df['train_loss'], 
                color=COLORS['train'], linewidth=2, label='Train', alpha=0.8)
        ax.plot(df['epoch'], df['val_loss'], 
                color=COLORS['val'], linewidth=2, label='Val', alpha=0.8)
        
        # Find best epoch, when a validation curve exists
        if df['val_loss'].notna().any():
            best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
            ax.axvline(x=best_epoch, color='gray', linestyle='--',
                       linewidth=1, alpha=0.3)
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title(display_name, fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        apply_axis_scale(ax, scale)

        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'all_models_grid_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {output_path}")


def print_summary_table(model_data):
    """
    Print a summary table of training results.
    
    Args:
        model_data: Dictionary mapping model names to DataFrames
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Best Epoch':<12} {'Train Loss':<15} {'Val Loss':<15} {'Epochs':<10}")
    print("-"*80)
    
    for model_name, df in model_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        if not df['val_loss'].notna().any():
            print(f"{display_name:<20} {'n/a':<12} {df['train_loss'].iloc[-1]:<15.6f} {'n/a':<15} {len(df):<10}")
            continue
        best_idx = df['val_loss'].idxmin()
        best_epoch = int(df.loc[best_idx, 'epoch'])
        best_train_loss = df.loc[best_idx, 'train_loss']
        best_val_loss = df.loc[best_idx, 'val_loss']
        total_epochs = len(df)
        
        print(f"{display_name:<20} {best_epoch:<12} {best_train_loss:<15.6f} {best_val_loss:<15.6f} {total_epochs:<10}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot loss curves for all ML models')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory containing model results')
    parser.add_argument('--output-dir', type=str, default='loss_curves_plots',
                        help='Directory to save all plots')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Override the model list (default: the published set)')
    parser.add_argument('--scale', type=str, default='loglog',
                        choices=['linear', 'semilogy', 'loglog'],
                        help="Axis scaling: 'loglog' (log epoch + log loss, default), "
                             "'semilogy' (log loss only), or 'linear'")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_names = args.models if args.models else MODEL_NAMES
    print(f"📊 Plotting loss curves for {len(model_names)} models...")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()
    
    # Load all model data
    model_data = {}
    for model_name in model_names:
        df = load_training_curves(model_name, args.results_dir)
        if df is not None:
            model_data[model_name] = df
    
    if not model_data:
        print("\n❌ No training curves found for any model!")
        return
    
    print(f"\n✅ Successfully loaded data for {len(model_data)} models\n")
    
    print(f"📐 Axis scale: {args.scale}")

    # Plot individual models
    print("Plotting individual model curves...")
    for model_name, df in model_data.items():
        plot_individual_model(df, model_name, output_dir, scale=args.scale)

    # Plot comparison charts
    print("\nPlotting comparison charts...")
    plot_all_models_comparison(model_data, output_dir, scale=args.scale)
    
    # Print summary table
    print_summary_table(model_data)
    
    print(f"\n✅ All plots saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
