"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Unified evaluator with FIXED normalization and complete metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
import json
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from .base_model import BenchmarkModel
from .metrics import (
    MetricsCalculator,
    ComputationalMetrics,
    TrainingMetrics,
    GeneralizationMetrics
)
from .bootstrap_metrics import BootstrapMetricsCalculator


class UnifiedEvaluator:
    """Evaluator that works with any model implementing BenchmarkModel interface."""
    
    def __init__(
        self,
        model_wrapper: BenchmarkModel,
        save_dir: str = 'results',
        enable_bootstrap: bool = False
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_wrapper: Model wrapper implementing BenchmarkModel
            save_dir: Directory to save results
            enable_bootstrap: Whether to compute bootstrap confidence intervals
        """
        self.wrapper = model_wrapper
        self.model = None
        self.save_dir = save_dir
        self.enable_bootstrap = enable_bootstrap
        
        # Get model info
        self.model_info = self.wrapper.get_model_info()
        self.model_name = self.model_info['name']
        
        # Initialize metrics trackers
        self.metrics_calc = MetricsCalculator(self.wrapper.device)
        self.train_metrics = TrainingMetrics()
        self.gen_metrics = GeneralizationMetrics()
        
        # Training history for visualization
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Store all split results
        self.split_results = {}
        
        # Results storage
        self.results = {
            'model_name': self.model_name,
            'model_info': self.model_info,
            'timestamp': datetime.now().isoformat(),
            'computational_metrics': {},
            'accuracy_metrics': {},
            'training_metrics': {},
            'generalization_metrics': {},
            'inference_metrics': {},
            'split_metrics': {}  # Store train/val/test separately
        }
        
        # Create save directory
        self.model_save_dir = os.path.join(save_dir, self.model_name.lower())
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def setup(self):
        """Setup the model and data loaders."""
        print(f"\nSetting up {self.model_name}...")
        
        # Load model
        self.model = self.wrapper.load_model()
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = self.wrapper.get_data_loaders()
        
        print(f"✓ Model loaded: {self.model.__class__.__name__}")
        print(f"✓ Train batches: {len(self.train_loader)}")
        print(f"✓ Val batches: {len(self.val_loader)}")
        print(f"✓ Test batches: {len(self.test_loader)}")
        
        # Initialize computational metrics
        sample_batch = next(iter(self.train_loader))
        sample_input, _ = self.wrapper.preprocess_batch(sample_batch)
        input_shape = sample_input.shape
        
        self.comp_metrics = ComputationalMetrics(self.model, input_shape)
    
    def evaluate_model_complexity(self):
        """Evaluate model computational complexity."""
        print(f"\nEvaluating computational complexity for {self.model_name}...")
        self.model.eval()
        # Parameter count
        param_metrics = self.comp_metrics.count_parameters()
        self.results['computational_metrics'].update(param_metrics)
        
        # FLOPs estimation
        try:
            flops_metrics = self.comp_metrics.estimate_flops()
            self.results['computational_metrics'].update(flops_metrics)
        except Exception as e:
            print(f"Warning: Could not estimate FLOPs: {e}")
        
        # Memory usage - add dummy method if missing
        try:
            # Simple memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                
                sample_batch = next(iter(self.train_loader))
                preprocess_result = self.wrapper.preprocess_batch(sample_batch)
                if len(preprocess_result) == 3:
                    sample_input, _, _ = preprocess_result
                else:
                    sample_input, _ = preprocess_result
                sample_input = sample_input.to(self.wrapper.device)
                
                _ = self.wrapper.forward(self.model, sample_input)
                torch.cuda.synchronize()
                
                peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
                self.results['computational_metrics']['peak_memory_gb'] = peak_memory
        except Exception as e:
            print(f"Warning: Could not measure memory usage: {e}")
        
        # Inference time with better statistics
        try:
            inference_metrics = self.measure_inference_time_detailed(num_runs=100)
            self.results['inference_metrics'] = inference_metrics
        except Exception as e:
            print(f"Warning: Could not measure inference time: {e}")
        
        return self.results['computational_metrics']
    
    def measure_inference_time_detailed(self, num_runs: int = 100) -> Dict[str, float]:
        """Measure detailed inference time statistics."""
        self.model.eval()
        times = []
        
        # Get a sample batch
        sample_batch = next(iter(self.test_loader))
        preprocess_result = self.wrapper.preprocess_batch(sample_batch)
        if len(preprocess_result) == 3:
            sample_input, _, _ = preprocess_result
        else:
            sample_input, _ = preprocess_result
        sample_input = sample_input.to(self.wrapper.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.wrapper.forward(self.model, sample_input)
        
        # Measure
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                _ = self.wrapper.forward(self.model, sample_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            'mean_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'median_inference_time_ms': np.median(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'p95_inference_time_ms': np.percentile(times, 95),
            'p99_inference_time_ms': np.percentile(times, 99)
        }
    
    def evaluate_on_dataset(self, split: str = 'test', verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model on a dataset split with FIXED normalization.
        
        CRITICAL FIX: Ensure predictions and targets are in the SAME scale for metrics.
        """
        # Select appropriate data loader
        if split == 'train':
            dataloader = self.train_loader
        elif split == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
        
        self.model.eval()
        
        # Get loss function and normalization functions
        criterion = self.wrapper.get_loss_function()
        normalize_fn, denormalize_fn = self.wrapper.get_normalization_functions()
        
        total_loss = 0
        total_samples = 0
        
        # Collect all predictions and targets for comprehensive metrics
        all_predictions = []
        all_targets = []
        
        if verbose:
            print(f"\nEvaluating on {split} set...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[{split}]", disable=not verbose):
                # Get raw data in ORIGINAL scale
                preprocess_result = self.wrapper.preprocess_batch(batch)
                if len(preprocess_result) == 3:
                    data, targets_original, _ = preprocess_result
                else:
                    data, targets_original = preprocess_result
                data = data.to(self.wrapper.device)
                targets_original = targets_original.to(self.wrapper.device)
                
                # Forward pass - model might output normalized or unnormalized
                outputs = self.wrapper.forward(self.model, data)
                
                # CRITICAL: Ensure both are in same scale for loss
                if normalize_fn is not None:
                    # Model likely outputs normalized, so normalize targets for loss
                    targets_for_loss = normalize_fn(targets_original)
                    loss = criterion(outputs, targets_for_loss)
                    
                    # Denormalize outputs for metrics computation
                    outputs_for_metrics = denormalize_fn(outputs)
                    targets_for_metrics = targets_original
                else:
                    # Model outputs in original scale
                    loss = criterion(outputs, targets_original)
                    outputs_for_metrics = outputs
                    targets_for_metrics = targets_original
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # For models like Transolver that use mixed surface+interior data,
                # split predictions to only evaluate on surface points
                if hasattr(self.wrapper, 'requires_surface_interior_split') and self.wrapper.requires_surface_interior_split():
                    # Split predictions and targets to get only surface points
                    outputs_surface, _ = self.wrapper.split_surface_interior(outputs_for_metrics)
                    targets_surface, _ = self.wrapper.split_surface_interior(targets_for_metrics)
                    
                    # Only collect surface points for metrics
                    all_predictions.append(outputs_surface.cpu())
                    all_targets.append(targets_surface.cpu())
                else:
                    # Collect all points for metrics (normal models)
                    all_predictions.append(outputs_for_metrics.cpu())
                    all_targets.append(targets_for_metrics.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute comprehensive metrics
        metrics = self.compute_comprehensive_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / total_samples
        
        # Save per-sample checkpoint CSV for statistical testing
        if split == 'test' and self.enable_bootstrap:
            print(f"\n💾 Saving per-sample checkpoint CSV for {split} set...")
            checkpoint_dir = os.path.join(self.model_save_dir, 'metrics', 'bootstrap')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Compute per-sample metrics
            pred_np = all_predictions.cpu().numpy()
            target_np = all_targets.cpu().numpy()
            
            per_sample_metrics = []
            for i in range(len(pred_np)):
                sample_pred = pred_np[i].flatten()
                sample_target = target_np[i].flatten()
                
                # Compute metrics for this sample
                mse = np.mean((sample_pred - sample_target) ** 2)
                mae = np.mean(np.abs(sample_pred - sample_target))
                rmse = np.sqrt(mse)
                
                # Relative errors
                target_norm = np.linalg.norm(sample_target)
                if target_norm > 0:
                    rel_l2 = np.linalg.norm(sample_pred - sample_target) / target_norm
                    rel_l1 = np.sum(np.abs(sample_pred - sample_target)) / np.sum(np.abs(sample_target))
                else:
                    rel_l2 = 0
                    rel_l1 = 0
                
                # R² for single sample (pearson correlation squared)
                if len(sample_pred) > 1:
                    ss_res = np.sum((sample_target - sample_pred) ** 2)
                    ss_tot = np.sum((sample_target - np.mean(sample_target)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                else:
                    r2 = 0
                
                per_sample_metrics.append({
                    'sample_idx': i,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'rel_l2': rel_l2,
                    'rel_l1': rel_l1,
                    'r2': r2
                })
            
            # Save to CSV
            df_checkpoint = pd.DataFrame(per_sample_metrics)
            checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_{split}_eval_checkpoint.csv')
            df_checkpoint.to_csv(checkpoint_path, index=False)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            print(f"  {len(df_checkpoint)} samples saved")
        
        # Compute bootstrap CIs for test set if enabled
        if split == 'test' and self.enable_bootstrap:
            from .bootstrap_metrics import BootstrapMetricsCalculator
            
            print("\n" + "="*80)
            print("Computing Bootstrap Confidence Intervals (B=2000)...")
            print("="*80)
            
            bootstrap_calc = BootstrapMetricsCalculator(
                n_bootstrap=2000,
                confidence_level=0.95,
                random_state=42,
                save_indices=True
            )
            
            bootstrap_results = bootstrap_calc.compute_metrics_with_ci(
                predictions=all_predictions.cpu().numpy(),
                targets=all_targets.cpu().numpy(),
                car_ids=None,
                stratify=False
            )
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_to_native(obj):
                """Recursively convert numpy types to Python native types."""
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return obj
            
            self.results['bootstrap_metrics'] = {
                key: convert_to_native(result.to_dict()) for key, result in bootstrap_results.items()
            }
            
            # Save bootstrap results
            bootstrap_dir = os.path.join(self.model_save_dir, 'bootstrap')
            os.makedirs(bootstrap_dir, exist_ok=True)
            
            # Save as JSON
            import json
            json_path = os.path.join(bootstrap_dir, 'bootstrap_ci.json')
            with open(json_path, 'w') as f:
                json.dump(self.results['bootstrap_metrics'], f, indent=2)
            
            # Save as CSV
            csv_data = []
            for metric_name, result_dict in self.results['bootstrap_metrics'].items():
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
                df.to_csv(os.path.join(bootstrap_dir, 'bootstrap_ci.csv'), index=False)
            
            print("\n✓ Bootstrap confidence intervals computed")
            print(bootstrap_calc.format_results_table(bootstrap_results))
            print(f"✓ Bootstrap results saved to {bootstrap_dir}/")
        
        # Store for generalization analysis
        self.gen_metrics.update_split_metrics(split, metrics)
        self.split_results[split] = metrics
        
        # Update main results for test set
        if split == 'test':
            self.results['accuracy_metrics'] = metrics
        
        # Store in split metrics
        self.results['split_metrics'][split] = metrics
        
        return metrics
    
    def compute_comprehensive_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics including relative errors."""
        # Reshape for proper computation
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        # Flatten for basic metrics
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        # Basic metrics
        mse = np.mean((pred_flat - target_flat) ** 2)
        mae = np.mean(np.abs(pred_flat - target_flat))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        r2 = 1 - (ss_res / (ss_tot))
        
        # Max error
        max_error = np.max(np.abs(pred_flat - target_flat))
        
        # # For each sample, compute ||pred - target|| / ||target||
        # if len(pred_np.shape) > 1:
        #     # Multi-dimensional case
        #     rel_l2 = np.mean(np.linalg.norm(pred_np - target_np, axis=1) / 
        #                     (np.linalg.norm(target_np, axis=1)))
            
        #     # Relative L1 Error (normalized) - per sample then averaged  
        #     rel_l1 = np.mean(np.sum(np.abs(pred_np - target_np), axis=1) / 
        #                     (np.sum(np.abs(target_np), axis=1)))
        # else:
        #     # 1D case
        #     rel_l2 = np.linalg.norm(pred_flat - target_flat) / (np.linalg.norm(target_flat))
        #     rel_l1 = np.sum(np.abs(pred_flat - target_flat)) / (np.sum(np.abs(target_flat)))

        # Global relative error
        rel_l2 = np.linalg.norm(pred_flat - target_flat) / np.linalg.norm(target_flat)
        rel_l1 = np.sum(np.abs(pred_flat - target_flat)) / np.sum(np.abs(target_flat))

        # Max relative errors (element-wise for debugging)
        elementwise_rel_errors = np.abs(pred_flat - target_flat) / (np.abs(target_flat))
        max_rel_error = np.max(elementwise_rel_errors)
        median_rel_error = np.median(elementwise_rel_errors)
        
        # Percentile errors  
        errors = np.abs(pred_flat - target_flat)
        p50_error = np.percentile(errors, 50)
        p90_error = np.percentile(errors, 90)
        p95_error = np.percentile(errors, 95)
        p99_error = np.percentile(errors, 99)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'max_error': float(max_error),
            'rel_l1': float(rel_l1),
            'rel_l2': float(rel_l2),
            'max_rel_error': float(max_rel_error),
            'median_rel_error': float(median_rel_error),
            'p50_error': float(p50_error),
            'p90_error': float(p90_error),
            'p95_error': float(p95_error),
            'p99_error': float(p99_error),
            'num_samples': len(pred_flat)
        }
    
    def evaluate_with_bootstrap_ci(self, 
                                   split: str = 'test', 
                                   n_bootstrap: int = 2000,
                                   confidence_level: float = 0.95,
                                   verbose: bool = True,
                                   stratify: bool = True) -> Dict:
        """
        Evaluate model with bootstrap confidence intervals.
        
        Uses stratified paired bootstrap over test designs (cars), resampling cars
        not individual points to preserve spatial correlation. Stratification by
        archetype (Estate/Notchback/Fastback) preserves category distribution.
        
        Args:
            split: Dataset split to evaluate on ('train', 'val', or 'test')
            n_bootstrap: Number of bootstrap resamples (default: 2000)
            confidence_level: Confidence level for CIs (default: 0.95)
            verbose: Print progress
            stratify: Stratify by archetype (E/N/F) to preserve category mix
        
        Returns:
            Dictionary with metrics and confidence intervals
        """
        # Select appropriate data loader
        if split == 'train':
            dataloader = self.train_loader
        elif split == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
        
        self.model.eval()
        
        # Get normalization functions
        normalize_fn, denormalize_fn = self.wrapper.get_normalization_functions()
        
        # Collect predictions, targets, and car IDs/names
        all_predictions = []
        all_targets = []
        all_car_ids = []
        all_car_names = []
        
        car_id = 0  # Unique ID for each sample (car)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COLLECTING PREDICTIONS FOR BOOTSTRAP")
            print(f"{'='*60}")
            print(f"Split: {split}")
            print(f"Starting inference on test set...")
            import sys
            sys.stdout.flush()
        
        # Try to get car names from dataset if available
        try:
            if hasattr(dataloader.dataset, 'dataset'):
                # It's a Subset
                base_dataset = dataloader.dataset.dataset
                indices = dataloader.dataset.indices
            else:
                base_dataset = dataloader.dataset
                indices = range(len(base_dataset))
            
            # Extract car names from VTK file paths
            if hasattr(base_dataset, 'vtk_files'):
                car_names_list = []
                for idx in indices:
                    vtk_path = base_dataset.vtk_files[idx]
                    car_name = os.path.basename(vtk_path).replace('.vtk', '')
                    car_names_list.append(car_name)
            else:
                car_names_list = None
                if verbose and stratify:
                    print("⚠️  Could not extract car names from dataset, stratification disabled")
                    stratify = False
        except:
            car_names_list = None
            if verbose and stratify:
                print("⚠️  Could not extract car names from dataset, stratification disabled")
                stratify = False
        
        with torch.no_grad():
            batch_idx = 0
            for batch in tqdm(dataloader, desc=f"[{split}] Bootstrap", disable=not verbose):
                # Get data
                preprocess_result = self.wrapper.preprocess_batch(batch)
                if len(preprocess_result) == 3:
                    data, targets_original, _ = preprocess_result
                else:
                    data, targets_original = preprocess_result
                data = data.to(self.wrapper.device)
                targets_original = targets_original.to(self.wrapper.device)
                
                # Forward pass
                outputs = self.wrapper.forward(self.model, data)
                
                # Denormalize if needed
                if normalize_fn is not None:
                    outputs_for_metrics = denormalize_fn(outputs)
                    targets_for_metrics = targets_original
                else:
                    outputs_for_metrics = outputs
                    targets_for_metrics = targets_original
                
                # Handle surface/interior split if needed
                if hasattr(self.wrapper, 'requires_surface_interior_split') and self.wrapper.requires_surface_interior_split():
                    outputs_surface, _ = self.wrapper.split_surface_interior(outputs_for_metrics)
                    targets_surface, _ = self.wrapper.split_surface_interior(targets_for_metrics)
                    
                    batch_predictions = outputs_surface.cpu().numpy()
                    batch_targets = targets_surface.cpu().numpy()
                else:
                    batch_predictions = outputs_for_metrics.cpu().numpy()
                    batch_targets = targets_for_metrics.cpu().numpy()
                
                # Assign car IDs and names (each sample in batch is a different car)
                batch_size = batch_predictions.shape[0]
                batch_car_ids = np.arange(car_id, car_id + batch_size)
                
                # Get car names for this batch
                if car_names_list is not None:
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = start_idx + batch_size
                    batch_car_names = car_names_list[start_idx:end_idx]
                else:
                    batch_car_names = [f"car_{i}" for i in batch_car_ids]
                
                car_id += batch_size
                batch_idx += 1
                
                all_predictions.append(batch_predictions)
                all_targets.append(batch_targets)
                all_car_ids.extend(batch_car_ids)
                all_car_names.extend(batch_car_names)
        
        # Concatenate all data
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_car_ids = np.array(all_car_ids)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Collected {len(all_car_ids)} car samples from {split} set")
            print(f"Total predictions shape: {all_predictions.shape}")
            print(f"Total targets shape: {all_targets.shape}")
            if stratify and car_names_list is not None:
                print(f"Running STRATIFIED bootstrap (by archetype) with B={n_bootstrap} resamples...")
            else:
                print(f"Running bootstrap with B={n_bootstrap} resamples...")
            print(f"{'='*60}")
            import sys
            sys.stdout.flush()
        
        # Run bootstrap
        bootstrap_calc = BootstrapMetricsCalculator(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=42,
            save_indices=True  # Save indices for reproducibility
        )
        
        bootstrap_results = bootstrap_calc.compute_metrics_with_ci(
            predictions=all_predictions,
            targets=all_targets,
            car_ids=all_car_ids,
            car_names=all_car_names if stratify else None,
            stratify=stratify
        )
        
        if verbose:
            print("\n" + bootstrap_calc.format_results_table(bootstrap_results))
        
        # Convert to dictionary format for storage
        results_dict = {}
        bootstrap_indices_saved = None
        
        for key, result in bootstrap_results.items():
            results_dict[f'{key}_mean'] = result.mean
            results_dict[f'{key}_ci_lower'] = result.ci_lower
            results_dict[f'{key}_ci_upper'] = result.ci_upper
            results_dict[f'{key}_std'] = result.std
            
            # Save bootstrap indices once (they're the same for all metrics)
            if bootstrap_indices_saved is None and result.bootstrap_indices is not None:
                bootstrap_indices_saved = result.bootstrap_indices
        
        # Save bootstrap indices to file for reproducibility
        if bootstrap_indices_saved is not None and hasattr(self, 'save_dir'):
            indices_path = os.path.join(self.save_dir, f'bootstrap_indices_{split}.json')
            try:
                import json
                with open(indices_path, 'w') as f:
                    json.dump({
                        'n_bootstrap': n_bootstrap,
                        'confidence_level': confidence_level,
                        'random_state': 42,
                        'stratified': stratify,
                        'n_cars': len(np.unique(all_car_ids)),
                        'car_ids': all_car_ids.tolist() if isinstance(all_car_ids, np.ndarray) else all_car_ids,
                        'car_names': all_car_names,
                        'bootstrap_indices': bootstrap_indices_saved,
                        'description': 'Bootstrap resampling indices (car IDs) for all B resamples. Use these to reproduce exact confidence intervals.'
                    }, f, indent=2)
                if verbose:
                    print(f"\n✅ Bootstrap indices saved to: {indices_path}")
                    print(f"   Use these indices to reproduce confidence intervals for any model on this test set.")
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Could not save bootstrap indices: {e}")
        
        # Add metadata
        results_dict['n_bootstrap'] = n_bootstrap
        results_dict['confidence_level'] = confidence_level
        results_dict['n_cars'] = len(np.unique(all_car_ids))
        results_dict['n_total_points'] = len(all_predictions.flatten())
        
        return results_dict
    
    def update_training_history(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Update training history for visualization."""
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['learning_rates'].append(lr)
    
    def evaluate_single_model(self, checkpoint_path: Optional[str] = None, enable_bootstrap: bool = False):
        """
        Evaluate a single model without training.
        
        Args:
            checkpoint_path: Path to model checkpoint
            enable_bootstrap: Whether to compute bootstrap CIs (slow, ~1-2 min)
        """
        print(f"\nEvaluating {self.model_name}...")
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model = self.wrapper.load_checkpoint(checkpoint_path, self.model)
        
        # Start timing
        eval_start = time.time()
        
        # Evaluate model complexity
        self.evaluate_model_complexity()
        
        # Evaluate on ALL splits and show results for each
        print("\n" + "="*60)
        print("EVALUATION ON ALL SPLITS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            metrics = self.evaluate_on_dataset(split, verbose=True)
            self._print_split_results(split, metrics)
        
        # Optionally compute bootstrap CIs for test set
        if enable_bootstrap:
            print("\n" + "="*60)
            print("BOOTSTRAP CONFIDENCE INTERVALS (Test Set)")
            print("="*60)
            print("⚠️  This may take 1-2 minutes...")
            bootstrap_metrics = self.evaluate_with_bootstrap_ci(
                split='test',
                n_bootstrap=2000,
                confidence_level=0.95,
                verbose=True
            )
            self.results['bootstrap_metrics'] = bootstrap_metrics
        
        # Compute generalization metrics
        self.results['generalization_metrics'] = self.gen_metrics.compute_generalization_gap()
        
        # Total evaluation time
        total_time = time.time() - eval_start
        self.results['total_evaluation_time_seconds'] = total_time
        
        # Save results
        self.save_results()
        
        # Print comprehensive summary
        self._print_evaluation_summary()
        
        return self.results
    
    def _print_split_results(self, split: str, metrics: Dict[str, float]):
        """Print results for a specific split."""
        print(f"\n{split.upper()} SET RESULTS:")
        print("-" * 40)
        print(f"  R² Score:        {metrics.get('r2', 0):.4f}")
        print(f"  MSE:             {metrics.get('mse', 0):.2f}")
        print(f"  MAE:             {metrics.get('mae', 0):.2f} Pa")
        print(f"  RMSE:            {metrics.get('rmse', 0):.2f} Pa")
        print(f"  Max Error:       {metrics.get('max_error', 0):.2f} Pa")
        print(f"  Relative L1:     {metrics.get('rel_l1', 0):.4f}")
        print(f"  Relative L2:     {metrics.get('rel_l2', 0):.4f}")
        print(f"  P95 Error:       {metrics.get('p95_error', 0):.2f} Pa")
    
    def _print_evaluation_summary(self):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print(f"COMPREHENSIVE EVALUATION SUMMARY: {self.model_name}")
        print("="*60)
        
        # Test set performance (primary)
        test_metrics = self.results['split_metrics'].get('test', {})
        print("\n TEST SET PERFORMANCE:")
        print(f"  R² Score:        {test_metrics.get('r2', 0):.4f}")
        print(f"  MSE:             {test_metrics.get('mse', 0):.2f}")
        print(f"  MAE:             {test_metrics.get('mae', 0):.2f} Pa")
        print(f"  RMSE:            {test_metrics.get('rmse', 0):.2f} Pa")
        print(f"  Max Error:       {test_metrics.get('max_error', 0):.2f} Pa")
        print(f"  Relative L1:     {test_metrics.get('rel_l1', 0):.4f}")
        print(f"  Relative L2:     {test_metrics.get('rel_l2', 0):.4f}")
        
        # Generalization
        gen = self.results['generalization_metrics']
        if gen:
            print("\n GENERALIZATION:")
            print(f"  Train R²:        {self.results['split_metrics'].get('train', {}).get('r2', 0):.4f}")
            print(f"  Val R²:          {self.results['split_metrics'].get('val', {}).get('r2', 0):.4f}")
            print(f"  Test R²:         {test_metrics.get('r2', 0):.4f}")
            print(f"  Train-Test Gap:  {gen.get('train_test_r2_gap', 0):.4f}")
        
        # Computational metrics
        comp = self.results['computational_metrics']
        if comp:
            print("\n COMPUTATIONAL COMPLEXITY:")
            print(f"  Parameters:      {comp.get('total_parameters', 0):,}")
            print(f"  Memory:          {comp.get('peak_memory_gb', 0):.2f} GB")
        
        # Inference metrics
        inf = self.results['inference_metrics']
        if inf:
            print("\n⚡ INFERENCE PERFORMANCE:")
            print(f"  Mean time:       {inf.get('mean_inference_time_ms', 0):.2f} ms")
            print(f"  Std time:        {inf.get('std_inference_time_ms', 0):.2f} ms")
            print(f"  P95 time:        {inf.get('p95_inference_time_ms', 0):.2f} ms")
            print(f"  Throughput:      {1000/max(inf.get('mean_inference_time_ms', 1), 1):.1f} samples/sec")
        
        # Total time
        print(f"\n Total Evaluation Time: {self.results.get('total_evaluation_time_seconds', 0):.1f} seconds")
        
        print("="*60)
    
    def train_and_evaluate(self):
        """Complete training and evaluation pipeline."""
        # Get training configuration
        train_config = self.wrapper.get_training_config()
        
        # Create trainer
        from .trainer import UnifiedTrainer
        trainer = UnifiedTrainer(
            model=self.model,
            wrapper=self.wrapper,
            evaluator=self,
            save_dir=self.model_save_dir
        )
        
        # Setup training components
        optimizer = self.wrapper.get_optimizer(self.model)
        scheduler = self.wrapper.get_scheduler(optimizer)
        
        # Train
        print(f"\nStarting training for {self.model_name}...")
        self.train_metrics.start_training()
        
        trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=train_config.get('epochs', 150),
            early_stopping_patience=train_config.get('early_stopping_patience', 20),
            gradient_clip=train_config.get('gradient_clip', 1.0)
        )
        
        # Evaluate on all splits
        print("\nFinal evaluation on all splits...")
        for split in ['train', 'val', 'test']:
            metrics = self.evaluate_on_dataset(split)
            self._print_split_results(split, metrics)
        
        # Compute training summary
        self.results['training_metrics'] = self.train_metrics.get_training_summary()
        
        # Compute generalization metrics
        self.results['generalization_metrics'] = self.gen_metrics.compute_generalization_gap()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save evaluation results."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert all results to serializable format
        serializable_results = convert_to_serializable(self.results)
        
        # Add training history if available
        if self.training_history['train_loss']:
            serializable_results['training_history'] = convert_to_serializable(self.training_history)
        
        # Save detailed JSON results
        json_path = os.path.join(
            self.model_save_dir,
            f"{self.model_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        def _json_default(o):
            # Safety net for anything convert_to_serializable missed (e.g. numpy
            # int64 dict keys/scalars) so results saving never aborts the run.
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4, default=_json_default)
        
        # Save comprehensive summary
        self._save_comprehensive_summary()
        
        print(f"\n✅ Results saved to {json_path}")
    
    def _save_comprehensive_summary(self):
        """Save comprehensive summary metrics for easy comparison."""
        # Get metrics from different splits
        train_metrics = self.results['split_metrics'].get('train', {})
        val_metrics = self.results['split_metrics'].get('val', {})
        test_metrics = self.results['split_metrics'].get('test', self.results.get('accuracy_metrics', {}))
        
        # Computational metrics
        comp_metrics = self.results['computational_metrics']
        inference_metrics = self.results['inference_metrics']
        training_metrics = self.results['training_metrics']
        gen_metrics = self.results['generalization_metrics']
        
        # Build comprehensive summary
        summary = {
            # Model Information
            'model_name': self.model_name,
            'timestamp': self.results['timestamp'],
            'evaluation_time_seconds': self.results.get('total_evaluation_time_seconds', 0),
            
            # Model Complexity
            'total_parameters': comp_metrics.get('total_parameters', 0),
            'trainable_parameters': comp_metrics.get('trainable_parameters', 0),
            'non_trainable_parameters': comp_metrics.get('non_trainable_parameters', 0),
            'parameter_size_mb': comp_metrics.get('parameter_size_mb', 0),
            'model_size_mb': comp_metrics.get('model_size_mb', 0),
            'flops_millions': comp_metrics.get('flops_millions', 0),
            'peak_memory_gb': comp_metrics.get('peak_memory_gb', 0),
            
            # Inference Performance
            'mean_inference_time_ms': inference_metrics.get('mean_inference_time_ms', 0),
            'std_inference_time_ms': inference_metrics.get('std_inference_time_ms', 0),
            'median_inference_time_ms': inference_metrics.get('median_inference_time_ms', 0),
            'p95_inference_time_ms': inference_metrics.get('p95_inference_time_ms', 0),
            'p99_inference_time_ms': inference_metrics.get('p99_inference_time_ms', 0),
            'throughput_samples_per_sec': 1000/max(inference_metrics.get('mean_inference_time_ms', 1), 1),
            
            # Training Metrics (if available)
            'training_time_hours': training_metrics.get('total_training_time_hours', 0),
            'epochs_trained': training_metrics.get('epochs_trained', 0),
            'best_epoch': training_metrics.get('best_epoch', 0),
            'early_stopped': training_metrics.get('early_stopped', False),
            'final_learning_rate': training_metrics.get('final_learning_rate', 0),
            
            # Train Set Performance
            'train_r2': train_metrics.get('r2', 0),
            'train_mse': train_metrics.get('mse', 0),
            'train_mae': train_metrics.get('mae', 0),
            'train_rmse': train_metrics.get('rmse', 0),
            'train_max_error': train_metrics.get('max_error', 0),
            'train_rel_l1': train_metrics.get('rel_l1', 0),
            'train_rel_l2': train_metrics.get('rel_l2', 0),
            'train_max_rel_error': train_metrics.get('max_rel_error', 0),
            'train_median_rel_error': train_metrics.get('median_rel_error', 0),
            'train_p95_error': train_metrics.get('p95_error', 0),
            'train_p99_error': train_metrics.get('p99_error', 0),
            'train_loss': train_metrics.get('loss', 0),
            'train_num_samples': train_metrics.get('num_samples', 0),
            
            # Validation Set Performance
            'val_r2': val_metrics.get('r2', 0),
            'val_mse': val_metrics.get('mse', 0),
            'val_mae': val_metrics.get('mae', 0),
            'val_rmse': val_metrics.get('rmse', 0),
            'val_max_error': val_metrics.get('max_error', 0),
            'val_rel_l1': val_metrics.get('rel_l1', 0),
            'val_rel_l2': val_metrics.get('rel_l2', 0),
            'val_max_rel_error': val_metrics.get('max_rel_error', 0),
            'val_median_rel_error': val_metrics.get('median_rel_error', 0),
            'val_p95_error': val_metrics.get('p95_error', 0),
            'val_p99_error': val_metrics.get('p99_error', 0),
            'val_loss': val_metrics.get('loss', 0),
            'val_num_samples': val_metrics.get('num_samples', 0),
            
            # Test Set Performance (Primary Results)
            'test_r2': test_metrics.get('r2', 0),
            'test_mse': test_metrics.get('mse', 0),
            'test_mae': test_metrics.get('mae', 0),
            'test_rmse': test_metrics.get('rmse', 0),
            'test_max_error': test_metrics.get('max_error', 0),
            'test_rel_l1': test_metrics.get('rel_l1', 0),
            'test_rel_l2': test_metrics.get('rel_l2', 0),
            'test_max_rel_error': test_metrics.get('max_rel_error', 0),
            'test_median_rel_error': test_metrics.get('median_rel_error', 0),
            'test_p50_error': test_metrics.get('p50_error', 0),
            'test_p90_error': test_metrics.get('p90_error', 0),
            'test_p95_error': test_metrics.get('p95_error', 0),
            'test_p99_error': test_metrics.get('p99_error', 0),
            'test_loss': test_metrics.get('loss', 0),
            'test_num_samples': test_metrics.get('num_samples', 0),
            
            # Generalization Metrics
            'train_val_r2_gap': gen_metrics.get('train_val_r2_gap', 0),
            'train_test_r2_gap': gen_metrics.get('train_test_r2_gap', 0),
            'val_test_r2_gap': gen_metrics.get('val_test_r2_gap', 0),
            'train_val_mse_ratio': gen_metrics.get('train_val_mse_ratio', 0),
            'train_test_mse_ratio': gen_metrics.get('train_test_mse_ratio', 0),
            'generalization_score': gen_metrics.get('generalization_score', 0),
            'overfitting_score': gen_metrics.get('overfitting_score', 0),
            
            # Performance Ratios and Rankings
            'efficiency_score': test_metrics.get('r2', 0) / max(comp_metrics.get('total_parameters', 1), 1) * 1e6,  # R²/param * 1M
            'speed_accuracy_ratio': test_metrics.get('r2', 0) / max(inference_metrics.get('mean_inference_time_ms', 1), 1),  # R²/ms
            'memory_efficiency': test_metrics.get('r2', 0) / max(comp_metrics.get('peak_memory_gb', 1), 1),  # R²/GB
        }
        
        # Append to comprehensive CSV
        csv_path = os.path.join(self.save_dir, 'comprehensive_benchmark_summary.csv')
        
        # Convert all values to ensure they're serializable
        summary = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                  for k, v in summary.items()}
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_new = pd.DataFrame([summary])
            
            # Ensure all columns match
            for col in df_new.columns:
                if col not in df_existing.columns:
                    df_existing[col] = 0
            for col in df_existing.columns:
                if col not in df_new.columns:
                    df_new[col] = 0
            
            # Reorder columns to match
            df_new = df_new[df_existing.columns]
            
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df = pd.DataFrame([summary])
            df.to_csv(csv_path, index=False)
        
        print(f"✅ Comprehensive summary saved to {csv_path}")
        
        # Also save the original simple summary for backward compatibility
        simple_summary = {
            'model_name': self.model_name,
            'timestamp': self.results['timestamp'],
            'total_parameters': comp_metrics.get('total_parameters', 0),
            'peak_memory_gb': comp_metrics.get('peak_memory_gb', 0),
            'mean_inference_time_ms': inference_metrics.get('mean_inference_time_ms', 0),
            'test_r2': test_metrics.get('r2', 0),
            'test_mse': test_metrics.get('mse', 0),
            'test_mae': test_metrics.get('mae', 0),
            'test_rmse': test_metrics.get('rmse', 0),
            'test_mean_rel_l1': test_metrics.get('rel_l1', 0),
            'training_time_hours': training_metrics.get('total_training_time_hours', 0),
            'train_test_r2_gap': gen_metrics.get('train_test_r2_gap', 0)
        }
        
        simple_csv_path = os.path.join(self.save_dir, 'benchmark_summary.csv')
        
        if os.path.exists(simple_csv_path):
            df_existing = pd.read_csv(simple_csv_path)
            df_new = pd.DataFrame([simple_summary])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(simple_csv_path, index=False)
        else:
            df = pd.DataFrame([simple_summary])
            df.to_csv(simple_csv_path, index=False)