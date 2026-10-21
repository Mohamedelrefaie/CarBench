#!/usr/bin/env python3
"""
Script to verify metric calculations used in the DrivAerNet benchmark.
This confirms that the metrics are computed correctly and provides examples.
"""

import numpy as np
import torch
from typing import Dict

def compute_metrics_manual(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Manually compute all metrics step by step for verification.
    
    Args:
        predictions: Predicted values (can be multi-dimensional)
        targets: Ground truth values (same shape as predictions)
    
    Returns:
        Dictionary of computed metrics
    """
    # Flatten arrays for basic metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    N = len(pred_flat)
    
    print(f"Total number of values (N): {N}")
    print(f"Shape before flattening: {predictions.shape}")
    print(f"Shape after flattening: {pred_flat.shape}")
    print()
    
    # 1. MSE - Mean Squared Error
    squared_errors = (pred_flat - target_flat) ** 2
    mse = np.mean(squared_errors)
    print(f"MSE calculation:")
    print(f"  Sum of squared errors: {np.sum(squared_errors):.4f}")
    print(f"  Number of values: {N}")
    print(f"  MSE = {np.sum(squared_errors):.4f} / {N} = {mse:.4f}")
    print()
    
    # 2. MAE - Mean Absolute Error
    absolute_errors = np.abs(pred_flat - target_flat)
    mae = np.mean(absolute_errors)
    print(f"MAE calculation:")
    print(f"  Sum of absolute errors: {np.sum(absolute_errors):.4f}")
    print(f"  MAE = {np.sum(absolute_errors):.4f} / {N} = {mae:.4f}")
    print()
    
    # 3. RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)
    print(f"RMSE = sqrt(MSE) = sqrt({mse:.4f}) = {rmse:.4f}")
    print()
    
    # 4. R² - Coefficient of determination
    mean_target = np.mean(target_flat)
    ss_tot = np.sum((target_flat - mean_target) ** 2)
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    print(f"R² calculation:")
    print(f"  Mean of targets: {mean_target:.4f}")
    print(f"  SS_tot (total sum of squares): {ss_tot:.4f}")
    print(f"  SS_res (residual sum of squares): {ss_res:.4f}")
    print(f"  R² = 1 - ({ss_res:.4f} / {ss_tot:.4f}) = {r2:.4f}")
    print()
    
    # 5. Relative L2 Error
    if len(predictions.shape) > 1:
        # Multi-dimensional case (e.g., batch of samples)
        # Compute per-sample L2 norms
        sample_errors = []
        for i in range(predictions.shape[0]):
            pred_norm = np.linalg.norm(predictions[i])
            error_norm = np.linalg.norm(predictions[i] - targets[i])
            if pred_norm > 1e-8:
                sample_errors.append(error_norm / pred_norm)
        rel_l2 = np.mean(sample_errors)
        print(f"Relative L2 (multi-dim, {predictions.shape[0]} samples):")
        print(f"  Per-sample relative errors: {sample_errors[:5]}...")
        print(f"  Mean relative L2: {rel_l2:.4f}")
    else:
        # 1D case
        pred_l2_norm = np.linalg.norm(pred_flat)
        error_l2_norm = np.linalg.norm(pred_flat - target_flat)
        rel_l2 = error_l2_norm / (pred_l2_norm + 1e-8)
        print(f"Relative L2 (1D case):")
        print(f"  ||predictions||_2: {pred_l2_norm:.4f}")
        print(f"  ||predictions - targets||_2: {error_l2_norm:.4f}")
        print(f"  Relative L2 = {error_l2_norm:.4f} / {pred_l2_norm:.4f} = {rel_l2:.4f}")
    print()
    
    # 6. Max Error
    max_error = np.max(absolute_errors)
    max_error_idx = np.argmax(absolute_errors)
    print(f"Maximum Error:")
    print(f"  Max error: {max_error:.4f}")
    print(f"  Occurs at index: {max_error_idx}")
    print(f"  Prediction at max: {pred_flat[max_error_idx]:.4f}")
    print(f"  Target at max: {target_flat[max_error_idx]:.4f}")
    print()
    
    # 7. Percentile Errors
    p50 = np.percentile(absolute_errors, 50)
    p90 = np.percentile(absolute_errors, 90)
    p95 = np.percentile(absolute_errors, 95)
    p99 = np.percentile(absolute_errors, 99)
    print(f"Percentile Errors:")
    print(f"  P50 (median): {p50:.4f}")
    print(f"  P90: {p90:.4f}")
    print(f"  P95: {p95:.4f}")
    print(f"  P99: {p99:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'rel_l2': rel_l2,
        'max_error': max_error,
        'p50_error': p50,
        'p90_error': p90,
        'p95_error': p95,
        'p99_error': p99,
        'n_total': N
    }


def test_with_synthetic_data():
    """Test metrics with synthetic data to verify calculations."""
    print("="*60)
    print("TEST 1: Simple 1D Case")
    print("="*60)
    
    # Simple 1D test case
    targets_1d = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    predictions_1d = np.array([95.0, 210.0, 295.0, 390.0, 510.0])
    
    print(f"Targets: {targets_1d}")
    print(f"Predictions: {predictions_1d}")
    print()
    
    metrics_1d = compute_metrics_manual(predictions_1d, targets_1d)
    
    print("\n" + "="*60)
    print("TEST 2: Multi-sample Case (like batch processing)")
    print("="*60)
    
    # Multi-sample case (3 samples, 5 points each)
    targets_2d = np.array([
        [100.0, 200.0, 300.0, 400.0, 500.0],
        [150.0, 250.0, 350.0, 450.0, 550.0],
        [200.0, 300.0, 400.0, 500.0, 600.0]
    ])
    
    predictions_2d = np.array([
        [95.0, 210.0, 295.0, 390.0, 510.0],
        [145.0, 245.0, 355.0, 460.0, 540.0],
        [205.0, 295.0, 405.0, 495.0, 605.0]
    ])
    
    print(f"Targets shape: {targets_2d.shape}")
    print(f"Predictions shape: {predictions_2d.shape}")
    print()
    
    metrics_2d = compute_metrics_manual(predictions_2d, targets_2d)
    
    print("\n" + "="*60)
    print("TEST 3: Realistic Pressure Data")
    print("="*60)
    
    # Simulate realistic pressure data (in Pascals)
    np.random.seed(42)
    n_samples = 10
    n_points = 1000
    
    # Generate realistic pressure values around atmospheric pressure variations
    # Mean around -100 Pa, std around 100 Pa (typical for car aerodynamics)
    targets_realistic = np.random.normal(-100, 100, (n_samples, n_points))
    # Add some systematic error and noise
    predictions_realistic = targets_realistic * 0.95 + np.random.normal(0, 20, (n_samples, n_points))
    
    print(f"Realistic data shape: {targets_realistic.shape}")
    print(f"Target pressure range: [{targets_realistic.min():.1f}, {targets_realistic.max():.1f}] Pa")
    print(f"Prediction pressure range: [{predictions_realistic.min():.1f}, {predictions_realistic.max():.1f}] Pa")
    print()
    
    metrics_realistic = compute_metrics_manual(predictions_realistic, targets_realistic)
    
    print("\n" + "="*60)
    print("SUMMARY OF ALL TESTS")
    print("="*60)
    
    print(f"\nTest 1 (Simple 1D): MSE={metrics_1d['mse']:.2f}, R²={metrics_1d['r2']:.4f}")
    print(f"Test 2 (Multi-sample): MSE={metrics_2d['mse']:.2f}, R²={metrics_2d['r2']:.4f}")
    print(f"Test 3 (Realistic): MSE={metrics_realistic['mse']:.2f}, R²={metrics_realistic['r2']:.4f}")
    
    # Verify against numpy/sklearn implementations
    print("\n" + "="*60)
    print("VERIFICATION AGAINST STANDARD IMPLEMENTATIONS")
    print("="*60)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Flatten for sklearn
    targets_flat = targets_realistic.flatten()
    predictions_flat = predictions_realistic.flatten()
    
    mse_sklearn = mean_squared_error(targets_flat, predictions_flat)
    mae_sklearn = mean_absolute_error(targets_flat, predictions_flat)
    r2_sklearn = r2_score(targets_flat, predictions_flat)
    
    print(f"Our MSE: {metrics_realistic['mse']:.6f}")
    print(f"Sklearn MSE: {mse_sklearn:.6f}")
    print(f"Difference: {abs(metrics_realistic['mse'] - mse_sklearn):.2e}")
    print()
    
    print(f"Our MAE: {metrics_realistic['mae']:.6f}")
    print(f"Sklearn MAE: {mae_sklearn:.6f}")
    print(f"Difference: {abs(metrics_realistic['mae'] - mae_sklearn):.2e}")
    print()
    
    print(f"Our R²: {metrics_realistic['r2']:.6f}")
    print(f"Sklearn R²: {r2_sklearn:.6f}")
    print(f"Difference: {abs(metrics_realistic['r2'] - r2_sklearn):.2e}")


if __name__ == "__main__":
    test_with_synthetic_data()