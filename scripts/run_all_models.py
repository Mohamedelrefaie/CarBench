#!/usr/bin/env python
"""
Script to run benchmarks for all available models.
"""

import sys
import os
import argparse
import subprocess
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def get_available_models():
    """Get list of available models from the models directory."""
    models_dir = os.path.join(project_root, 'models')
    models = []
    
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Check if wrapper.py exists
            wrapper_path = os.path.join(item_path, 'wrapper.py')
            if os.path.exists(wrapper_path):
                models.append(item)
    
    return sorted(models)


def run_single_model(model_name, dataset_path, subset_dir, save_dir):
    """Run benchmark for a single model."""
    cmd = [
        sys.executable,
        os.path.join(project_root, 'scripts', 'run_single_model.py'),
        model_name,
        '--dataset-path', dataset_path,
        '--subset-dir', subset_dir,
        '--save-dir', save_dir
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {model_name}:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception running {model_name}: {e}")
        return False


def generate_comparison_report(save_dir):
    """Generate comparison report from individual model results."""
    summary_path = os.path.join(save_dir, 'benchmark_summary.csv')
    
    if not os.path.exists(summary_path):
        print("No results found to compare")
        return
    
    # Load summary data
    df = pd.read_csv(summary_path)
    
    # Sort by test MSE (lower is better)
    df = df.sort_values('test_mse')
    
    # Create detailed comparison report
    report_path = os.path.join(save_dir, f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DrivAerNet++ Benchmark Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        # Best model by accuracy
        f.write("BEST MODEL BY ACCURACY (MSE):\n")
        f.write(f"  {df.iloc[0]['model_name']} - MSE: {df.iloc[0]['test_mse']:.6f}\n\n")
        
        # Best model by efficiency
        fastest_idx = df['mean_inference_time_ms'].idxmin()
        f.write("FASTEST MODEL:\n")
        f.write(f"  {df.loc[fastest_idx, 'model_name']} - {df.loc[fastest_idx, 'mean_inference_time_ms']:.2f} ms\n\n")
        
        # Full rankings
        f.write("FULL RANKINGS:\n")
        f.write("-"*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Detailed metrics
        f.write("DETAILED METRICS:\n")
        f.write("-"*80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"\n{row['model_name']}:\n")
            f.write(f"  Accuracy:    MSE={row['test_mse']:.6f}, MAE={row['test_mae']:.6f}, R²={row['test_r2']:.4f}\n")
            f.write(f"  Efficiency:  {row['total_parameters']:,} params, {row['gflops']:.2f} GFLOPs, {row['mean_inference_time_ms']:.2f} ms\n")
            f.write(f"  Training:    {row['training_time_hours']:.2f} hours\n")
            f.write(f"  Gen. Gap:    {row['train_test_mse_gap']:.6f}\n")
    
    print(f"\nComparison report saved to: {report_path}")
    
    # Also create a visual comparison plot
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # MSE comparison
        ax1.bar(df['model_name'], df['test_mse'])
        ax1.set_ylabel('Test MSE')
        ax1.set_title('Model Accuracy (lower is better)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        ax2.bar(df['model_name'], df['mean_inference_time_ms'])
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Speed (lower is better)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Parameters comparison
        ax3.bar(df['model_name'], df['total_parameters'] / 1e6)
        ax3.set_ylabel('Parameters (M)')
        ax3.set_title('Model Size')
        ax3.tick_params(axis='x', rotation=45)
        
        # Training time comparison
        ax4.bar(df['model_name'], df['training_time_hours'])
        ax4.set_ylabel('Training Time (hours)')
        ax4.set_title('Training Cost')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'comparison_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300)
        print(f"Comparison plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Could not create comparison plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for all models')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to DrivAerNet++ dataset')
    parser.add_argument('--subset-dir', type=str, required=True,
                        help='Path to train/val/test splits')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to benchmark (default: all)')
    
    args = parser.parse_args()
    
    # Get models to benchmark
    if args.models:
        models = args.models
    else:
        models = get_available_models()
    
    print("="*60)
    print("DrivAerNet++ Full Benchmark Suite")
    print("="*60)
    print(f"Models to benchmark: {', '.join(models)}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run each model
    successful_models = []
    failed_models = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Benchmarking {model}...")
        print("="*60)
        
        success = run_single_model(model, args.dataset_path, args.subset_dir, args.save_dir)
        
        if success:
            successful_models.append(model)
        else:
            failed_models.append(model)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUITE COMPLETED")
    print("="*60)
    print(f"Successful: {len(successful_models)} models - {', '.join(successful_models)}")
    if failed_models:
        print(f"Failed: {len(failed_models)} models - {', '.join(failed_models)}")
    
    # Generate comparison report
    if successful_models:
        print("\nGenerating comparison report...")
        generate_comparison_report(args.save_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
