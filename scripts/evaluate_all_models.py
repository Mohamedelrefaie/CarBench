#!/usr/bin/env python3
"""
Evaluate All Models Script
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu

This script runs evaluation for all available models sequentially and ranks them
based on their relative L2 error values for easy comparison.
"""

import os
import sys
import argparse
import subprocess
import json
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

# Model configurations with their expected checkpoint paths
MODELS = {
    'pointnet': {
        'config': 'models/pointnet/config.yaml',
        'checkpoint': 'results_backup/pointnet/checkpoints/best_checkpoint.pth',
        'description': 'PointNet - Basic point cloud processing'
    },
    'pointnet_large': {
        'config': 'models/pointnet_large/config.yaml', 
        'checkpoint': 'results_backup/pointnetlarge/checkpoints/best_checkpoint.pth',
        'description': 'PointNet Large - Enhanced PointNet'
    },
    'regdgcnn': {
        'config': 'models/regdgcnn/config.yaml',
        'checkpoint': 'results_backup/regdgcnn/checkpoints/best_checkpoint.pth', 
        'description': 'RegDGCNN - Dynamic Graph CNN'
    },
    'point_transformer': {
        'config': 'models/point_transformer/config.yaml',
        'checkpoint': 'results_backup/pointtransformer/checkpoints/best_checkpoint.pth',
        'description': 'Point Transformer - Attention-based'
    },
    'pointmae': {
        'config': 'models/pointmae/config.yaml',
        'checkpoint': 'results_backup/pointmae/checkpoints/best_checkpoint.pth',
        'description': 'PointMAE - Masked Autoencoder'
    },
    'transolver': {
        'config': 'models/transolver/config.yaml',
        'checkpoint': 'results_backup/transolver/checkpoints/best_checkpoint.pth',
        'description': 'Transolver - Physics-informed transformer'
    },
    'transolver_plus': {
        'config': 'models/transolver_plus/config.yaml',
        'checkpoint': 'results_backup/transolver_plus/checkpoints/best_checkpoint.pth',
        'description': 'Transolver++ - Enhanced physics transformer'
    },
    'transolver_large': {
        'config': 'models/transolver_large/config.yaml',
        'checkpoint': 'results_backup/transolver_large/checkpoints/best_checkpoint.pth',
        'description': 'Transolver Large - Scalable physics transformer'
    },
    'fno': {
        'config': 'models/fno/config.yaml',
        'checkpoint': 'results_backup/neuraloperator/checkpoints/best_checkpoint.pth',
        'description': 'FNO - Fourier Neural Operator'
    },
    # 'abupt': {
    #     'config': 'models/abupt/config.yaml',
    #     'checkpoint': 'results/abupt/checkpoints/best_checkpoint.pth',
    #     'description': 'AB-UPT - Attention-based Unified Point Transformer'
    # }
}

def check_checkpoint_exists(checkpoint_path: str) -> bool:
    """Check if checkpoint file exists"""
    return os.path.exists(checkpoint_path)

def run_evaluation(model_name: str, model_info: Dict, args: argparse.Namespace) -> Optional[Dict]:
    """Run evaluation for a single model and return results"""
    
    checkpoint_path = model_info['checkpoint']
    config_path = model_info['config']
    
    # Check if checkpoint exists
    if not check_checkpoint_exists(checkpoint_path):
        print(f"❌ Checkpoint not found for {model_name}: {checkpoint_path}")
        return None
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"❌ Config not found for {model_name}: {config_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"🚀 Evaluating {model_name.upper()}")
    print(f"📋 Description: {model_info['description']}")
    print(f"⚙️  Config: {config_path}")
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        'python', 'scripts/train_and_analyze.py', model_name,
        '--dataset-path', args.dataset_path,
        '--subset-dir', args.subset_dir, 
        '--save-dir', args.save_dir,
        '--config', config_path,
        '--evaluate-only',
        '--checkpoint', checkpoint_path
    ]
    
    # Add optional batch size
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    
    # Add bootstrap flag if enabled
    if args.enable_bootstrap:
        cmd.append('--enable-bootstrap')
    
    # Set CUDA devices
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🎮 CUDA devices: {args.cuda_devices}")
    
    start_time = time.time()
    
    try:
        # Run evaluation with live output streaming
        print(f"\n🔄 Starting evaluation (live output)...\n")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print without extra newline
            output_lines.append(line)
        
        # Wait for process to complete
        returncode = process.wait(timeout=args.timeout)
        output = ''.join(output_lines)
        
        duration = time.time() - start_time
        
        if returncode == 0:
            print(f"\n✅ {model_name} completed successfully in {duration:.1f}s")
            
            # Try to parse metrics from output
            metrics = parse_metrics_from_output(output)
            if metrics:
                print(f"📊 Parsed metrics: {metrics}")
                metrics['duration'] = duration
                metrics['status'] = 'success'
                return metrics
            else:
                print(f"⚠️  Could not parse metrics from {model_name} output")
                # Save output for debugging
                debug_path = f"/tmp/{model_name}_output.txt"
                with open(debug_path, 'w') as f:
                    f.write(output)
                print(f"🔍 Full output saved to {debug_path} for debugging")
                return {'status': 'success_no_metrics', 'duration': duration}
        else:
            print(f"\n❌ {model_name} failed with return code {returncode}")
            return {'status': 'failed', 'duration': duration, 'error': 'See output above'}
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {model_name} timed out after {args.timeout}s")
        process.kill()
        return {'status': 'timeout', 'duration': args.timeout}
    except Exception as e:
        print(f"\n💥 {model_name} crashed: {str(e)}")
        return {'status': 'crashed', 'duration': time.time() - start_time, 'error': str(e)}

def parse_metrics_from_output(output: str) -> Optional[Dict]:
    """Parse evaluation metrics from command output"""
    metrics = {}
    
    # Look for the specific "Final Test Performance:" section from train_and_analyze.py
    lines = output.split('\n')
    
    # Look for the final test performance section
    in_final_section = False
    
    for line in lines:
        line = line.strip()
        
        # Look for the start of final test performance section
        if 'Final Test Performance:' in line:
            in_final_section = True
            continue
        
        # If we're in the final section, parse metrics
        if in_final_section:
            # Stop if we hit the end delimiter
            if line.startswith('====') or line == '':
                if metrics:  # Only break if we've collected some metrics
                    break
            
            # Parse each metric line
            if 'R² Score:' in line or 'R2 Score:' in line:
                try:
                    value = line.split(':')[1].strip()
                    metrics['r2'] = float(value)
                except:
                    pass
            elif 'MSE:' in line:
                try:
                    value = line.split(':')[1].strip()
                    metrics['mse'] = float(value)
                except:
                    pass
            elif 'MAE:' in line:
                try:
                    value = line.split(':')[1].strip().replace(' Pa', '')
                    metrics['mae'] = float(value)
                except:
                    pass
            elif 'RMSE:' in line:
                try:
                    value = line.split(':')[1].strip().replace(' Pa', '')
                    metrics['rmse'] = float(value)
                except:
                    pass
            elif 'Relative L2:' in line:
                try:
                    value = line.split(':')[1].strip()
                    metrics['relative_l2'] = float(value)
                except:
                    pass
            elif 'Max Error:' in line:
                try:
                    value = line.split(':')[1].strip().replace(' Pa', '')
                    metrics['max_error'] = float(value)
                except:
                    pass
    
    # Fallback: Look for patterns anywhere in the output
    if not metrics:
        import re
        for metric_name in ['mse', 'mae', 'rmse', 'relative_l2', 'r2']:
            # Look for patterns like "MSE: 123.45" or "metric_name: value"
            pattern = rf'{metric_name}[:\s]+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)'
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])  # Take the last match
                except:
                    pass
    
    return metrics if metrics else None

def create_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a results table from evaluation results"""
    
    data = []
    
    for model_name, result in results.items():
        if result and result.get('status') == 'success':
            row = {
                'Model': model_name,
                'Description': MODELS[model_name]['description'],
                'Status': '✅ Success',
                'Duration (s)': f"{result.get('duration', 0):.1f}",
                'MSE': result.get('mse', 'N/A'),
                'MAE': result.get('mae', 'N/A'), 
                'RMSE': result.get('rmse', 'N/A'),
                'Relative L2': result.get('relative_l2', 'N/A'),
                'R²': result.get('r2', 'N/A')
            }
        elif result:
            row = {
                'Model': model_name,
                'Description': MODELS[model_name]['description'],
                'Status': f"❌ {result.get('status', 'Unknown')}",
                'Duration (s)': f"{result.get('duration', 0):.1f}",
                'MSE': 'N/A',
                'MAE': 'N/A',
                'RMSE': 'N/A', 
                'Relative L2': 'N/A',
                'R²': 'N/A'
            }
        else:
            row = {
                'Model': model_name,
                'Description': MODELS[model_name]['description'],
                'Status': '❌ Not Found',
                'Duration (s)': '0.0',
                'MSE': 'N/A',
                'MAE': 'N/A',
                'RMSE': 'N/A',
                'Relative L2': 'N/A', 
                'R²': 'N/A'
            }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by Relative L2 (lower is better) - put successful models first
    def sort_key(row):
        if row['Status'] == '✅ Success' and row['Relative L2'] != 'N/A':
            return (0, float(row['Relative L2']))  # Successful models, sorted by rel_l2
        elif row['Status'] == '✅ Success':
            return (1, 0)  # Successful but no metrics
        else:
            return (2, 0)  # Failed models at the end
    
    df['_sort_key'] = df.apply(sort_key, axis=1)
    df = df.sort_values('_sort_key').drop('_sort_key', axis=1).reset_index(drop=True)
    
    return df

def save_results(results: Dict, df: pd.DataFrame, args: argparse.Namespace):
    """Save results to files"""
    
    # Create output directory
    output_dir = Path(args.save_dir) / 'evaluation_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results as JSON
    json_path = output_dir / f'detailed_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary table as CSV
    csv_path = output_dir / f'model_comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    # Save summary table as markdown (simple format to avoid tabulate dependency)
    md_path = output_dir / f'model_comparison_{timestamp}.md'
    with open(md_path, 'w') as f:
        f.write("# Model Evaluation Results\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Create simple markdown table manually
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
        
        f.write("\n\n## Notes\n")
        f.write("- Models are ranked by Relative L2 error (lower is better)\n")
        f.write("- ✅ indicates successful evaluation\n")
        f.write("- ❌ indicates failed evaluation or missing checkpoint\n")
    
    print(f"\n📊 Results saved to:")
    print(f"   📄 JSON: {json_path}")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📄 Markdown: {md_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate all models and rank by performance')
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "PressureVTK", "Dataverse")),
        help='Path to dataset'
    )
    
    parser.add_argument(
        '--subset-dir', 
        type=str,
        default=os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "train_val_test_splits")),
        help='Path to train/val/test splits'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str, 
        default='results',
        help='Results directory'
    )
    
    parser.add_argument(
        '--cuda-devices',
        type=str,
        default='0,1,2,3',
        help='CUDA devices to use (comma-separated)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for evaluation (optional)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,  # 1 hour
        help='Timeout per model in seconds'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip models with missing checkpoints instead of reporting them'
    )
    
    parser.add_argument(
        '--enable-bootstrap',
        action='store_true',
        help='Enable bootstrap confidence intervals (B=2000, adds ~1-2 min per model)'
    )
    
    args = parser.parse_args()
    
    # Determine which models to evaluate
    models_to_eval = args.models if args.models else list(MODELS.keys())
    
    print("🎯 Model Evaluation Pipeline")
    print("=" * 50)
    print(f"📂 Dataset: {args.dataset_path}")
    print(f"📂 Splits: {args.subset_dir}")
    print(f"💾 Save dir: {args.save_dir}")
    print(f"🎮 CUDA devices: {args.cuda_devices}")
    print(f"📦 Models to evaluate: {', '.join(models_to_eval)}")
    print(f"⏰ Timeout per model: {args.timeout}s")
    if args.enable_bootstrap:
        print(f"🔬 Bootstrap CIs: ENABLED (B=2000, adds ~1-2 min per model)")
    else:
        print(f"🔬 Bootstrap CIs: DISABLED")
    print("=" * 50)
    
    results = {}
    total_models = len(models_to_eval)
    
    for i, model_name in enumerate(models_to_eval, 1):
        if model_name not in MODELS:
            print(f"⚠️  Unknown model: {model_name}")
            continue
            
        print(f"\n📊 Progress: {i}/{total_models}")
        
        model_info = MODELS[model_name]
        result = run_evaluation(model_name, model_info, args)
        
        if result is None and args.skip_missing:
            print(f"⏭️  Skipping {model_name} (missing checkpoint)")
            continue
            
        results[model_name] = result
    
    # Create and display results table
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS - MODEL RANKING")
    print(f"{'='*80}")
    
    df = create_results_table(results)
    print(df.to_string(index=False))
    
    # Save results
    save_results(results, df, args)
    
    # Summary statistics
    successful = sum(1 for r in results.values() if r and r.get('status') == 'success')
    total = len(results)
    
    print(f"\n📈 Summary:")
    print(f"   ✅ Successful evaluations: {successful}/{total}")
    print(f"   ❌ Failed evaluations: {total - successful}/{total}")
    
    if successful > 0:
        # Find best model by relative L2
        best_model = None
        best_rel_l2 = float('inf')
        
        for model_name, result in results.items():
            if (result and result.get('status') == 'success' and 
                'relative_l2' in result and isinstance(result['relative_l2'], (int, float))):
                if result['relative_l2'] < best_rel_l2:
                    best_rel_l2 = result['relative_l2']
                    best_model = model_name
        
        if best_model:
            print(f"\n🏆 Best performing model: {best_model}")
            print(f"   🎯 Relative L2 error: {best_rel_l2:.6f}")
            print(f"   📋 Description: {MODELS[best_model]['description']}")

if __name__ == '__main__':
    main()
