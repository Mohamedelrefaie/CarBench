"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Bootstrap confidence intervals for model evaluation metrics.

Implements stratified paired bootstrap over test designs (car geometries) with B=2000 resamples.
Bootstrap is performed by resampling cars (not individual points) to maintain spatial correlation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class BootstrapResult:
    """Container for bootstrap results with confidence intervals"""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    bootstrap_indices: Optional[List[List[int]]] = None  # Indices for all B resamples
    
    def __repr__(self):
        return f"{self.mean:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
    
    def to_dict(self):
        result = {
            'mean': self.mean,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'std': self.std
        }
        if self.bootstrap_indices is not None:
            result['bootstrap_indices'] = self.bootstrap_indices
        return result


class BootstrapMetricsCalculator:
    """
    Calculate metrics with bootstrap confidence intervals.
    
    Bootstrap procedure:
    1. Resample car designs WITH replacement (B=2000 times)
    2. For each resample, calculate metric across all points in those cars
    3. Compute 95% CI from bootstrap distribution
    
    This preserves spatial correlation within each car design.
    """
    
    def __init__(self, 
                 n_bootstrap: int = 2000,
                 confidence_level: float = 0.95,
                 random_state: Optional[int] = 42,
                 save_indices: bool = True):
        """
        Args:
            n_bootstrap: Number of bootstrap resamples (default: 2000)
            confidence_level: Confidence level for intervals (default: 0.95)
            random_state: Random seed for reproducibility
            save_indices: Whether to save bootstrap indices for reproducibility (default: True)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.save_indices = save_indices
    
    def compute_metrics_with_ci(self,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                car_ids: Optional[np.ndarray] = None,
                                car_names: Optional[List[str]] = None,
                                stratify: bool = True) -> Dict[str, BootstrapResult]:
        """
        Compute metrics with bootstrap confidence intervals.
        
        Args:
            predictions: Predicted values, shape [n_samples, n_points] or [n_total_points]
            targets: Ground truth values, same shape as predictions
            car_ids: Array of car IDs for each sample, shape [n_samples]
                    If None, treats each sample as a separate car
            car_names: List of car names for stratification (e.g., "E_S_WWC_WM_005")
                      First letter indicates archetype: E=Estate, N=Notchback, F=Fastback
            stratify: If True, perform stratified bootstrap by archetype
        
        Returns:
            Dictionary mapping metric names to BootstrapResult objects
        """
        # Ensure numpy arrays
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        
        # Handle different input shapes
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            targets = targets.reshape(-1, 1)
        
        n_samples = predictions.shape[0]
        
        # Create car IDs if not provided
        if car_ids is None:
            car_ids = np.arange(n_samples)
        else:
            car_ids = np.asarray(car_ids)
        
        # Get unique cars
        unique_cars = np.unique(car_ids)
        n_cars = len(unique_cars)
        
        # Determine if we should stratify
        use_stratified = stratify and car_names is not None and len(car_names) == n_samples
        
        if use_stratified:
            print(f"\n{'='*60}")
            print(f"STRATIFIED BOOTSTRAP")
            print(f"{'='*60}")
            print(f"Unique cars: {n_cars}")
            print(f"Total samples: {n_samples}")
            print(f"Bootstrap resamples: {self.n_bootstrap}")
            print(f"Extracting archetypes from car names...")
            import sys
            sys.stdout.flush()
            return self._compute_stratified_bootstrap(predictions, targets, car_ids, car_names)
        else:
            print(f"\n{'='*60}")
            print(f"REGULAR BOOTSTRAP")
            print(f"{'='*60}")
            print(f"Unique cars: {n_cars}")
            print(f"Total samples: {n_samples}")
            print(f"Bootstrap resamples: {self.n_bootstrap}")
            import sys
            sys.stdout.flush()
        
        # Compute point estimate (on full data)
        print(f"Computing point estimates on full data...")
        import sys
        sys.stdout.flush()
        point_metrics = self._compute_single_metrics(predictions, targets)
        print(f"Point estimates computed. Starting {self.n_bootstrap} bootstrap iterations...")
        sys.stdout.flush()
        
        # Bootstrap resampling
        bootstrap_metrics = {key: [] for key in point_metrics.keys()}
        bootstrap_indices = [] if self.save_indices else None  # Save car IDs for each resample
        
        print(f"\nStarting bootstrap loop NOW...")
        import sys
        sys.stdout.flush()
        
        for b in tqdm(range(self.n_bootstrap), desc="Bootstrap resampling", unit="iter", ncols=80):
            # Resample cars with replacement
            resampled_car_ids = self.rng.choice(unique_cars, size=n_cars, replace=True)
            
            # Save indices for reproducibility
            if self.save_indices:
                bootstrap_indices.append(resampled_car_ids.tolist())
            
            # Get indices for all points in resampled cars
            mask = np.isin(car_ids, resampled_car_ids)
            
            # Handle cases where same car is sampled multiple times
            resampled_preds = []
            resampled_targets = []
            
            for car_id in resampled_car_ids:
                car_mask = car_ids == car_id
                resampled_preds.append(predictions[car_mask])
                resampled_targets.append(targets[car_mask])
            
            # Concatenate all points from resampled cars
            resampled_preds = np.concatenate(resampled_preds, axis=0)
            resampled_targets = np.concatenate(resampled_targets, axis=0)
            
            # Compute metrics on bootstrap sample
            boot_metrics = self._compute_single_metrics(resampled_preds, resampled_targets)
            
            for key, value in boot_metrics.items():
                bootstrap_metrics[key].append(value)
        
        # Compute confidence intervals
        alpha = 1 - self.confidence_level
        results = {}
        
        for key in point_metrics.keys():
            boot_values = np.array(bootstrap_metrics[key])
            
            # Remove NaN/Inf values
            boot_values = boot_values[np.isfinite(boot_values)]
            
            if len(boot_values) < 100:
                warnings.warn(f"Only {len(boot_values)} valid bootstrap samples for {key}")
            
            # Compute percentile CI
            ci_lower = np.percentile(boot_values, 100 * alpha / 2)
            ci_upper = np.percentile(boot_values, 100 * (1 - alpha / 2))
            
            results[key] = BootstrapResult(
                mean=point_metrics[key],
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                std=np.std(boot_values),
                bootstrap_indices=bootstrap_indices if self.save_indices else None
            )
        
        return results
    
    def _compute_stratified_bootstrap(self,
                                     predictions: np.ndarray,
                                     targets: np.ndarray,
                                     car_ids: np.ndarray,
                                     car_names: List[str]) -> Dict[str, BootstrapResult]:
        """
        Compute bootstrap CIs with stratification by car archetype.
        
        Stratified bootstrap preserves the category distribution (Estate/Notchback/Fastback)
        in each bootstrap resample, leading to more accurate confidence intervals.
        
        Args:
            predictions: [n_samples, n_points]
            targets: [n_samples, n_points]
            car_ids: [n_samples] - unique ID for each car
            car_names: [n_samples] - car name strings (e.g., "E_S_WWC_WM_005")
        
        Returns:
            Dictionary of BootstrapResult objects
        """
        # Extract archetype from car names (first letter: E/N/F)
        archetypes = np.array([name[0] if name else 'U' for name in car_names])
        
        # Group cars by archetype
        unique_cars = np.unique(car_ids)
        car_to_archetype = {}
        car_to_indices = {}
        
        for car_id in unique_cars:
            car_mask = car_ids == car_id
            car_indices = np.where(car_mask)[0]
            car_archetype = archetypes[car_indices[0]]  # All samples from same car have same archetype
            
            car_to_archetype[car_id] = car_archetype
            car_to_indices[car_id] = car_indices
        
        # Group cars by archetype
        archetype_cars = {}
        for car_id, archetype in car_to_archetype.items():
            if archetype not in archetype_cars:
                archetype_cars[archetype] = []
            archetype_cars[archetype].append(car_id)
        
        # Count cars per archetype
        archetype_counts = {arch: len(cars) for arch, cars in archetype_cars.items()}
        
        print(f"\n{'='*60}")
        print(f"Archetype distribution:")
        for arch, count in sorted(archetype_counts.items()):
            arch_name = {'E': 'Estate', 'N': 'Notchback', 'F': 'Fastback'}.get(arch, 'Unknown')
            print(f"  {arch_name} ({arch}): {count} cars")
        print(f"{'='*60}")
        import sys
        sys.stdout.flush()
        
        # Compute point estimate (on full data)
        print(f"Computing point estimates on full data...")
        sys.stdout.flush()
        point_metrics = self._compute_single_metrics(predictions, targets)
        print(f"Point estimates computed. Starting {self.n_bootstrap} stratified bootstrap iterations...")
        sys.stdout.flush()
        
        # Bootstrap resampling with stratification
        bootstrap_metrics = {key: [] for key in point_metrics.keys()}
        bootstrap_indices = [] if self.save_indices else None  # Save car IDs for each resample
        
        print(f"\nStarting bootstrap loop NOW...")
        import sys
        sys.stdout.flush()
        
        for b in tqdm(range(self.n_bootstrap), desc="Stratified bootstrap", unit="iter", ncols=80):
            # For each archetype, resample cars with replacement
            resampled_car_ids = []
            
            for archetype, cars in archetype_cars.items():
                n_cars_in_archetype = len(cars)
                # Sample same number of cars as in original (with replacement)
                sampled_cars = self.rng.choice(cars, size=n_cars_in_archetype, replace=True)
                resampled_car_ids.extend(sampled_cars)
            
            # Save indices for reproducibility
            if self.save_indices:
                bootstrap_indices.append([int(cid) for cid in resampled_car_ids])
            
            # Collect all samples from resampled cars
            resampled_preds = []
            resampled_targets = []
            
            for car_id in resampled_car_ids:
                car_indices = car_to_indices[car_id]
                resampled_preds.append(predictions[car_indices])
                resampled_targets.append(targets[car_indices])
            
            # Concatenate all points from resampled cars
            resampled_preds = np.concatenate(resampled_preds, axis=0)
            resampled_targets = np.concatenate(resampled_targets, axis=0)
            
            # Compute metrics on bootstrap sample
            boot_metrics = self._compute_single_metrics(resampled_preds, resampled_targets)
            
            for key, value in boot_metrics.items():
                bootstrap_metrics[key].append(value)
        
        # Compute confidence intervals
        alpha = 1 - self.confidence_level
        results = {}
        
        for key in point_metrics.keys():
            boot_values = np.array(bootstrap_metrics[key])
            
            # Remove NaN/Inf values
            boot_values = boot_values[np.isfinite(boot_values)]
            
            if len(boot_values) < 100:
                warnings.warn(f"Only {len(boot_values)} valid bootstrap samples for {key}")
            
            # Compute percentile CI
            ci_lower = np.percentile(boot_values, 100 * alpha / 2)
            ci_upper = np.percentile(boot_values, 100 * (1 - alpha / 2))
            
            results[key] = BootstrapResult(
                mean=point_metrics[key],
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                std=np.std(boot_values),
                bootstrap_indices=bootstrap_indices if self.save_indices else None
            )
        
        return results
    
    def _compute_single_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics for a single dataset (or bootstrap sample).
        
        Args:
            predictions: Predicted values [n_samples, n_points] or [n_points]
            targets: Ground truth values, same shape
        
        Returns:
            Dictionary of metric values
        """
        # Flatten if needed for point-wise metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Basic error metrics
        errors = pred_flat - target_flat
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2
        
        mse = np.mean(sq_errors)
        mae = np.mean(abs_errors)
        rmse = np.sqrt(mse)
        max_error = np.max(abs_errors)
        
        # R² score
        ss_res = np.sum(sq_errors)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'max_error': max_error
        }
        
        # Relative errors (if data has 2D structure with points per sample)
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            # Compute per-sample relative errors
            l2_errors = np.linalg.norm(predictions - targets, ord=2, axis=-1)
            l2_norms = np.linalg.norm(targets, ord=2, axis=-1)
            
            # Avoid division by zero
            valid_mask = l2_norms > 1e-10
            if np.sum(valid_mask) > 0:
                rel_l2 = np.mean(l2_errors[valid_mask] / l2_norms[valid_mask])
            else:
                rel_l2 = 0.0
            
            # Relative L1
            l1_errors = np.linalg.norm(predictions - targets, ord=1, axis=-1)
            l1_norms = np.linalg.norm(targets, ord=1, axis=-1)
            
            valid_mask = l1_norms > 1e-10
            if np.sum(valid_mask) > 0:
                rel_l1 = np.mean(l1_errors[valid_mask] / l1_norms[valid_mask])
            else:
                rel_l1 = 0.0
            
            metrics['rel_l2'] = rel_l2
            metrics['rel_l1'] = rel_l1
        
        # Percentile errors
        metrics['p50_error'] = np.percentile(abs_errors, 50)
        metrics['p90_error'] = np.percentile(abs_errors, 90)
        metrics['p95_error'] = np.percentile(abs_errors, 95)
        metrics['p99_error'] = np.percentile(abs_errors, 99)
        
        return metrics
    
    def format_results_table(self, results: Dict[str, BootstrapResult]) -> str:
        """
        Format results as a nice table string.
        
        Args:
            results: Dictionary of metric name to BootstrapResult
        
        Returns:
            Formatted table string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("METRICS WITH 95% CONFIDENCE INTERVALS (Bootstrap B=2000)")
        lines.append("=" * 70)
        lines.append(f"{'Metric':<20} {'Value':<15} {'95% CI':<30}")
        lines.append("-" * 70)
        
        # Order metrics nicely
        metric_order = ['r2', 'mse', 'mae', 'rmse', 'rel_l2', 'rel_l1', 
                       'max_error', 'p50_error', 'p90_error', 'p95_error', 'p99_error']
        
        for metric in metric_order:
            if metric in results:
                result = results[metric]
                lines.append(f"{metric:<20} {result.mean:<15.6f} [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        
        # Add any remaining metrics
        for metric, result in results.items():
            if metric not in metric_order:
                lines.append(f"{metric:<20} {result.mean:<15.6f} [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def export_results(self, results: Dict[str, BootstrapResult], format: str = 'paper') -> str:
        """
        Export results in specific format for papers/reports.
        
        Args:
            results: Dictionary of metric name to BootstrapResult
            format: 'paper' for publication format, 'detailed' for full info
        
        Returns:
            Formatted string
        """
        if format == 'paper':
            # Format like: R² 0.9593 [0.9571, 0.9611]
            lines = []
            metric_names = {
                'r2': 'R²',
                'mse': 'MSE',
                'mae': 'MAE',
                'rmse': 'RMSE',
                'rel_l2': 'Rel. L2',
                'rel_l1': 'Rel. L1'
            }
            
            for key, name in metric_names.items():
                if key in results:
                    r = results[key]
                    lines.append(f"{name}: {r.mean:.4f} [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
            
            return "\n".join(lines)
        
        else:  # detailed
            return self.format_results_table(results)


def stratified_bootstrap_by_category(predictions_by_category: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                     n_bootstrap: int = 2000,
                                     confidence_level: float = 0.95,
                                     random_state: Optional[int] = 42) -> Dict[str, Dict[str, BootstrapResult]]:
    """
    Perform stratified bootstrap by vehicle category (Estate, Fastback, MPV, SUV, Wagon).
    
    Args:
        predictions_by_category: Dict mapping category name to (predictions, targets, car_ids)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for CIs
        random_state: Random seed
    
    Returns:
        Dictionary mapping category name to metrics with CIs
    """
    calculator = BootstrapMetricsCalculator(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state
    )
    
    results_by_category = {}
    
    for category, (preds, targets, car_ids) in predictions_by_category.items():
        print(f"\nBootstrap for category: {category}")
        results = calculator.compute_metrics_with_ci(preds, targets, car_ids)
        results_by_category[category] = results
    
    return results_by_category


def cross_category_bootstrap(predictions_by_category: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                             n_bootstrap: int = 2000,
                             confidence_level: float = 0.95,
                             random_state: Optional[int] = 42) -> Dict[str, BootstrapResult]:
    """
    Compute overall metrics with CIs across all categories combined.
    
    Args:
        predictions_by_category: Dict mapping category name to (predictions, targets, car_ids)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for CIs
        random_state: Random seed
    
    Returns:
        Dictionary of overall metrics with CIs
    """
    # Combine all data
    all_preds = []
    all_targets = []
    all_car_ids = []
    
    car_id_offset = 0
    for category, (preds, targets, car_ids) in predictions_by_category.items():
        all_preds.append(preds)
        all_targets.append(targets)
        # Offset car IDs to ensure uniqueness across categories
        all_car_ids.append(car_ids + car_id_offset)
        car_id_offset += np.max(car_ids) + 1
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_car_ids = np.concatenate(all_car_ids, axis=0)
    
    calculator = BootstrapMetricsCalculator(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state
    )
    
    print("\nBootstrap for overall (all categories)")
    return calculator.compute_metrics_with_ci(all_preds, all_targets, all_car_ids)
