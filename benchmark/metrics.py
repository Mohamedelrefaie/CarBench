# benchmark/metrics.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Comprehensive metrics calculation for regression tasks including error metrics, computational complexity, 
training dynamics, and generalization gaps.
"""

import torch
import numpy as np
from typing import Dict

class MetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
    
    def compute(self):
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        all_targets = torch.cat(self.targets, dim=0).numpy()
        
        metrics = {
            'mse': np.mean((all_preds - all_targets) ** 2),
            'mae': np.mean(np.abs(all_preds - all_targets)),
            'rmse': np.sqrt(np.mean((all_preds - all_targets) ** 2)),
            'r2': 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2),
            'max_error': np.max(np.abs(all_preds - all_targets))
        }
        
        # Add relative L1 and L2 errors - EXACTLY like your train.py
        if all_preds.ndim == 2:  # [batch, num_points]
            # Relative L2 error
            l2_errors = np.linalg.norm(all_preds - all_targets, ord=2, axis=-1)
            l2_norms = np.linalg.norm(all_targets, ord=2, axis=-1)
            metrics['rel_l2'] = np.mean(l2_errors / l2_norms)
            
            # Relative L1 error
            l1_errors = np.linalg.norm(all_preds - all_targets, ord=1, axis=-1)
            l1_norms = np.linalg.norm(all_targets, ord=1, axis=-1)
            metrics['rel_l1'] = np.mean(l1_errors / l1_norms)
        
        # Add percentile errors
        errors = np.abs(all_preds - all_targets)
        metrics['p50_error'] = np.percentile(errors, 50)  # Median error
        metrics['p90_error'] = np.percentile(errors, 90)
        metrics['p95_error'] = np.percentile(errors, 95)
        metrics['p99_error'] = np.percentile(errors, 99)
        
        return metrics

class ComputationalMetrics:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total_parameters': total, 'trainable_parameters': trainable}
    
    def estimate_flops(self):
        # Simplified FLOPs estimation
        return {'flops': 0, 'gflops': 0}
    
    def measure_inference_time(self, num_runs=100):
        import time
        times = []
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Measure
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = self.model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        times = np.array(times)
        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'min_inference_time_ms': np.min(times) * 1000,
            'max_inference_time_ms': np.max(times) * 1000,
            'p50_inference_time_ms': np.percentile(times, 50) * 1000,
            'p90_inference_time_ms': np.percentile(times, 90) * 1000,
            'p99_inference_time_ms': np.percentile(times, 99) * 1000,
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }

class TrainingMetrics:
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.epoch_losses = []
        self.learning_rates = []
        
    def start_training(self):
        import time
        self.start_time = time.time()
    
    def end_epoch(self, epoch, loss, lr):
        import time
        self.epoch_times.append(time.time())
        self.epoch_losses.append(loss)
        self.learning_rates.append(lr)
    
    def get_training_summary(self):
        if not self.epoch_times:
            return {}
        import time
        total_time = self.epoch_times[-1] - self.start_time if self.start_time else 0
        
        summary = {
            'total_training_time_hours': total_time / 3600,
            'total_training_time_minutes': total_time / 60,
            'total_epochs': len(self.epoch_times),
            'best_loss': min(self.epoch_losses) if self.epoch_losses else 0,
            'best_loss_epoch': self.epoch_losses.index(min(self.epoch_losses)) + 1 if self.epoch_losses else 0,
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else 0,
            'final_learning_rate': self.learning_rates[-1] if self.learning_rates else 0
        }
        
        # Add epoch duration statistics
        if len(self.epoch_times) > 1:
            epoch_durations = np.diff([self.start_time] + self.epoch_times)
            summary['mean_epoch_time_seconds'] = np.mean(epoch_durations)
            summary['std_epoch_time_seconds'] = np.std(epoch_durations)
        
        return summary

class GeneralizationMetrics:
    def __init__(self):
        self.metrics = {}
    
    def update_split_metrics(self, split, metrics):
        self.metrics[split] = metrics
    
    def compute_generalization_gap(self):
        gaps = {}
        
        # Train-Val gap
        if 'train' in self.metrics and 'val' in self.metrics:
            for key in ['mse', 'mae', 'rel_l1', 'rel_l2']:
                if key in self.metrics['train'] and key in self.metrics['val']:
                    gaps[f'train_val_{key}_gap'] = self.metrics['val'][key] - self.metrics['train'][key]
            
            if 'r2' in self.metrics['train'] and 'r2' in self.metrics['val']:
                gaps['train_val_r2_gap'] = self.metrics['train']['r2'] - self.metrics['val']['r2']
        
        # Train-Test gap
        if 'train' in self.metrics and 'test' in self.metrics:
            for key in ['mse', 'mae', 'rel_l1', 'rel_l2']:
                if key in self.metrics['train'] and key in self.metrics['test']:
                    gaps[f'train_test_{key}_gap'] = self.metrics['test'][key] - self.metrics['train'][key]
            
            if 'r2' in self.metrics['train'] and 'r2' in self.metrics['test']:
                gaps['train_test_r2_gap'] = self.metrics['train']['r2'] - self.metrics['test']['r2']
        
        return gaps