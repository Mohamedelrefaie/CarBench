#!/usr/bin/env python3
"""
Fixed single car inference script with consistent evaluation methodology.
Ensures fair comparison between all models by evaluating on same surface points.
Saves separate VTK files for predictions and error maps.
"""

import torch
import pyvista as pv
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import os
import yaml
import argparse
import sys
import logging
from typing import Dict, Tuple, Optional
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project setup
project_root = os.getcwd()
sys.path.append(project_root)

# Import data generation functions from transolver
sys.path.append(os.path.join(project_root, 'models', 'transolver'))
from data_loader import generate_interior_points, compute_sdf_polydata

MODEL_CHECKPOINT_PATHS = {
    'regdgcnn': os.path.join(project_root, 'results/regdgcnn/checkpoints/best_checkpoint.pth'),
    'pointmae': os.path.join(project_root, 'results/pointmae/checkpoints/best_checkpoint.pth'), 
    'pointnet': os.path.join(project_root, 'results/pointnet/checkpoints/best_checkpoint.pth'),
    'pointnet_large': os.path.join(project_root, 'results/pointnetlarge/checkpoints/best_checkpoint.pth'),
    'fno': os.path.join(project_root, 'results/neuraloperator/checkpoints/best_checkpoint.pth'),
    'point_transformer': os.path.join(project_root, 'results/pointtransformer/checkpoints/best_checkpoint.pth'),
    'transolver': os.path.join(project_root, 'results/transolver/checkpoints/best_checkpoint.pth'),
    'transolver_plus': os.path.join(project_root, 'results/transolver_plus/checkpoints/best_checkpoint.pth'),
    'transolver_large': os.path.join(project_root, 'results/transolver_large/checkpoints/best_checkpoint.pth'),
    'abupt': os.path.join(project_root, 'models/surface-abupt-drivaernet/pressure_abupt_checkpoint.th'),
}

# Normalization constants (unified across all models)
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25


def load_wrapper_and_model(model_name: str, checkpoint_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, object, Dict]:
    """Load model wrapper and checkpoint with proper configuration."""
    import importlib.util
    
    # Special handling for AB-UPT
    if model_name.lower() == 'abupt':
        return load_abupt_model(checkpoint_path, device)
    
    # Load config
    config_path = os.path.join(project_root, 'models', model_name, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = {}
        logger.warning(f"No config found at {config_path}, using defaults")
    
    # Set dummy paths for dataset (required by some wrappers)
    config['dataset_path'] = '/tmp/dummy'
    config['subset_dir'] = '/tmp/dummy'
    
    # Clear cached modules that might interfere
    modules_to_clear = [k for k in sys.modules.keys() if k == 'model' or k.startswith('model.')]
    for module_name in modules_to_clear:
        del sys.modules[module_name]
    
    # Add model directory to Python path
    model_dir = os.path.join(project_root, 'models', model_name)
    original_path = sys.path.copy()
    
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    try:
        # Load wrapper
        wrapper_path = os.path.join(project_root, 'models', model_name, 'wrapper.py')
        if not os.path.exists(wrapper_path):
            raise FileNotFoundError(f"Wrapper not found at {wrapper_path}")
        
        spec = importlib.util.spec_from_file_location(f"{model_name}_wrapper", wrapper_path)
        wrapper_module = importlib.util.module_from_spec(spec)
        wrapper_module.__path__ = [model_dir]
        spec.loader.exec_module(wrapper_module)
        
        # Find wrapper class
        wrapper_class = None
        for name, obj in wrapper_module.__dict__.items():
            if name.endswith('Wrapper') and hasattr(obj, '__bases__'):
                wrapper_class = obj
                break
        
        if wrapper_class is None:
            raise ValueError(f"No wrapper class found for {model_name}")
        
        # Initialize wrapper and load model
        wrapper = wrapper_class(config)
        model = wrapper.load_model()
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        if hasattr(wrapper, 'load_checkpoint'):
            try:
                model = wrapper.load_checkpoint(checkpoint_path, model)
            except RuntimeError as e:
                if "size mismatch" in str(e) and model_name == "transolver_plus":
                    logger.warning(f"⚠️  TRANSOLVER_PLUS CHECKPOINT MISMATCH:")
                    logger.warning(f"   The existing checkpoint was trained with SDF features (10D)")
                    logger.warning(f"   But current config uses surface-only features (6D)")
                    logger.warning(f"   Please retrain transolver_plus with surface-only configuration")
                    logger.warning(f"   Skipping transolver_plus for now...")
                    raise ValueError(f"Checkpoint incompatible - needs retraining with surface-only config")
                else:
                    raise
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Handle DataParallel prefix
            model_is_dp = isinstance(model, torch.nn.DataParallel)
            state_has_module = any(k.startswith('module.') for k in state_dict.keys())
            
            if state_has_module and not model_is_dp:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif not state_has_module and model_is_dp:
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        model = model.to(device)
        
        logger.info(f"Successfully loaded {model_name} model")
        
        return model, wrapper, config
        
    finally:
        # Restore original Python path
        sys.path = original_path


def load_abupt_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, object, Dict]:
    """Load AB-UPT model with special handling for its architecture."""
    logger.info("Loading AB-UPT model...")
    
    # Add AB-UPT src directory to path
    abupt_dir = os.path.join(project_root, 'models', 'surface-abupt-drivaernet')
    src_dir = os.path.join(abupt_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from surface_only_model import SurfaceOnlyABUPT
        
        # Initialize model
        model = SurfaceOnlyABUPT().to(device).eval()
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"AB-UPT checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict=checkpoint["state_dict"])
        
        logger.info("Successfully loaded AB-UPT model")
        
        # Create dummy wrapper and config for consistency
        class DummyWrapper:
            def __init__(self):
                pass
        
        config = {"model_type": "abupt"}
        
        return model, DummyWrapper(), config
        
    except Exception as e:
        logger.error(f"Failed to load AB-UPT model: {e}")
        raise


def sample_consistent_surface_points(mesh: pv.PolyData, n_points: int = 10000, 
                                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample consistent surface points that ALL models will be evaluated on.
    This ensures fair comparison between models.
    """
    if seed is not None:
        np.random.seed(seed)
    
    surface_points = mesh.points.astype(np.float32)
    surface_pressure = mesh.point_data['p'].astype(np.float32)
    
    if len(surface_points) > n_points:
        # Randomly sample surface points
        indices = np.random.choice(len(surface_points), n_points, replace=False)
        sampled_points = surface_points[indices]
        sampled_pressure = surface_pressure[indices]
    else:
        # Use all available points
        sampled_points = surface_points
        sampled_pressure = surface_pressure
    
    logger.info(f"📍 CONSISTENT EVALUATION: Using {len(sampled_points)} surface points for ALL models")
    logger.info(f"Pressure range: [{sampled_pressure.min():.1f}, {sampled_pressure.max():.1f}] Pa")
    
    return sampled_points, sampled_pressure


def prepare_transolver_data(consistent_surface_points: np.ndarray, consistent_surface_pressures: np.ndarray,
                           mesh: pv.PolyData, model_name: str = "transolver", sample_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare Transolver data for all models:
    - ALL Transolver models (transolver, transolver_large, transolver_plus): surface-only (6D features: x,y,z,nx,ny,nz)
    
    All models now use consistent 6D surface-only features for fair comparison.
    """
    # ALL TRANSOLVER MODELS: Surface-only mode with 6D features
    logger.info(f"🎯 {model_name.upper()} DATA: {len(consistent_surface_points)} surface points (surface-only mode)")
    logger.info(f"Surface-only mode with 6D features [x,y,z,nx,ny,nz] - consistent across all Transolver models")
    
    return consistent_surface_points, consistent_surface_pressures


def run_inference_transolver(model: torch.nn.Module, wrapper: object, points: np.ndarray, 
                           mesh: pv.PolyData, config: Dict, ground_truth_pressures: Optional[np.ndarray] = None, 
                           device: str = 'cuda', model_name: str = "transolver") -> np.ndarray:
    """Run inference for Transolver models with EXACT training feature computation.
    
    All Transolver models now use surface-only mode with 6D features [x,y,z,nx,ny,nz].
    """
    
    # Ensure points are numpy array
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    # ALL TRANSOLVER MODELS: Surface-only mode with 6D features
    logger.info("Computing 6D features [x, y, z, nx, ny, nz] for surface-only models")
    
    # Use EXACTLY the same normal computation as training
    surface_with_normals = mesh.compute_normals(
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True
    )
    
    # Get normals at the query points using nearest neighbor
    from sklearn.neighbors import NearestNeighbors
    surface_points = mesh.points.astype(np.float64)
    surface_normals = surface_with_normals.point_data['Normals'].astype(np.float32)
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(surface_points)
    _, indices = nbrs.kneighbors(points.astype(np.float64))
    normals = surface_normals[indices.flatten()]
    
    # Create features in EXACT same order as training [x, y, z, nx, ny, nz]
    features = np.concatenate([
        points,                    # [N, 3] - coordinates
        normals                   # [N, 3] - surface normals
    ], axis=1)                    # [N, 6] total
    
    logger.info(f"Final feature shape: {features.shape} (6D for surface-only model)")
    
    # Log feature statistics for debugging
    logger.info(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Convert to tensor with EXACT same preprocessing as training
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    if ground_truth_pressures is not None:
        # Normalize pressure EXACTLY like training (model expects normalized inputs)
        normalized_pressure = (ground_truth_pressures - PRESSURE_MEAN) / PRESSURE_STD
        targets = torch.tensor(normalized_pressure, dtype=torch.float32)
        logger.info(f"Normalized GT pressure range: [{normalized_pressure.min():.3f}, {normalized_pressure.max():.3f}]")
    else:
        targets = torch.zeros(features_tensor.shape[0], dtype=torch.float32)
    
    # Create batch EXACTLY like training DataLoader (with batch dimension)
    batch = (features_tensor.unsqueeze(0), targets.unsqueeze(0))
    
    # Use wrapper preprocessing (should match training exactly)
    data, _ = wrapper.preprocess_batch(batch)
    data = data.to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = wrapper.forward(model, data)
    
    # Denormalize predictions to physical units (Pa) for meaningful evaluation
    normalize_fn, denormalize_fn = wrapper.get_normalization_functions()
    if denormalize_fn is not None:
        prediction = denormalize_fn(prediction)
        logger.info(f"Denormalized predictions to physical units")
    
    result = prediction.cpu().numpy().flatten()
    
    logger.info(f"Final prediction range: [{result.min():.1f}, {result.max():.1f}] Pa (physical units)")
    if ground_truth_pressures is not None:
        logger.info(f"Ground truth range: [{ground_truth_pressures.min():.1f}, {ground_truth_pressures.max():.1f}] Pa (physical units)")
    
    return result


def run_inference_standard(model: torch.nn.Module, wrapper: object, points: np.ndarray, 
                         ground_truth_pressures: Optional[np.ndarray] = None, 
                         device: str = 'cuda') -> np.ndarray:
    """Run inference for standard models (non-Transolver)."""
    # Convert to tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    if points_tensor.dim() == 2:
        points_tensor = points_tensor.unsqueeze(0)
    
    if points_tensor.shape[-1] == 3:
        points_tensor = points_tensor.transpose(1, 2)
    
    # Create targets
    if ground_truth_pressures is not None:
        targets = torch.tensor(ground_truth_pressures, dtype=torch.float32).unsqueeze(0)
    else:
        targets = torch.zeros((1, points_tensor.shape[-1]), dtype=torch.float32)
    
    # Preprocess and run inference
    batch = (points_tensor, targets)
    data, _ = wrapper.preprocess_batch(batch)
    data = data.to(device)
    
    with torch.no_grad():
        prediction = wrapper.forward(model, data)
    
    # Denormalize if needed
    normalize_fn, denormalize_fn = wrapper.get_normalization_functions()
    if denormalize_fn is not None:
        prediction = denormalize_fn(prediction)
    
    return prediction.cpu().numpy().flatten()


def run_inference_abupt(model: torch.nn.Module, consistent_surface_points: np.ndarray, 
                       consistent_surface_pressures: np.ndarray, mesh: pv.PolyData,
                       device: str = 'cuda') -> np.ndarray:
    """Run inference for AB-UPT model with proper data preparation."""
    # Import AB-UPT specific components
    abupt_dir = os.path.join(project_root, 'models', 'surface-abupt-drivaernet')
    sys.path.insert(0, os.path.join(abupt_dir, 'src'))
    
    try:
        from surface_abupt_collator import SurfaceOnlyAbuptCollator
        from drivaernet_dataset import DrivAerNetStats
        # from surface_only_model import SurfaceOnlyABUPT  # loaded elsewhere
    except ImportError as e:
        logger.error(f"Failed to import AB-UPT components: {e}")
        raise
    
    logger.info("🚀 AB-UPT: Preparing data for inference")
    
    # Convert surface points to tensor
    surface_position = torch.from_numpy(consistent_surface_points).float()
    surface_pressure = torch.from_numpy(consistent_surface_pressures).float()
    
    # Ensure pressure has correct shape [N, 1]
    if surface_pressure.dim() == 1:
        surface_pressure = surface_pressure.unsqueeze(1)
    
    # Create dummy wall shear stress
    wallshearstress = torch.zeros(len(surface_position), 3)
    
    # Create raw sample matching AB-UPT format
    raw_sample = dict(
        surface_position_vtp=surface_position,
        surface_pressure=surface_pressure,
        surface_wallshearstress=wallshearstress,
    )
    
    # Create minimal dataset wrapper for collator (following tutorial)
    class VTKDatasetWrapper:
        def __init__(self):
            self.stats = DrivAerNetStats()
        
        def get_normalization_stats(self):
            return self.stats
    
    dataset = VTKDatasetWrapper()
    stats = dataset.get_normalization_stats()
    
    # Initialize collator for single-car evaluation:
    # - Use ALL sampled points as anchors to predict pressure exactly at those points
    # - Disable queries to simplify and avoid partial coverage
    num_geometry_points = min(65536, len(surface_position))
    num_surface_anchor_points = len(surface_position)
    num_geometry_supernodes = min(16384, num_geometry_points)
    
    collator = SurfaceOnlyAbuptCollator(
        num_geometry_points=num_geometry_points,
        num_surface_anchor_points=num_surface_anchor_points,
        num_geometry_supernodes=num_geometry_supernodes,
        use_query_positions=False,  # IMPORTANT: no queries for single-car pointwise eval
        dataset=dataset,
    )
    
    logger.info(f"AB-UPT: Using {num_geometry_points} geometry points, {num_surface_anchor_points} anchors")
    
    # Process through collator
    batch = collator([raw_sample])
    
    # Keep CPU copy for positions to remap order later, move a clone to GPU for model
    batch_gpu = {key: value.to(device) for key, value in batch.items()}
    
    # Extract targets (not used for loss, but shapes are useful)
    target_surface_anchor_pressure = batch_gpu.pop("surface_anchor_pressure")
    _ = batch_gpu.pop("surface_anchor_wallshearstress")
    # Queries are disabled; pop safely if present
    batch_gpu.pop("surface_query_pressure", None)
    batch_gpu.pop("surface_query_wallshearstress", None)
    
    logger.info(f"AB-UPT: Predicting {len(target_surface_anchor_pressure)} anchors only")
    
    # Run inference (following tutorial)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16), torch.no_grad():
        prediction = model(**batch_gpu)
    
    # Extract predictions at anchor points
    pred_anchor_pressure = prediction["surface_anchor_pressure"]  # [N,1]
    full_surface_prediction = pred_anchor_pressure
    
    logger.info(f"AB-UPT: Generated {len(full_surface_prediction)} total predictions")
    
    # Convert to numpy
    predictions = full_surface_prediction.cpu().numpy().flatten()
    
    # Log ranges pre-denorm
    logger.info(f"AB-UPT: Raw prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    logger.info(f"AB-UPT: Ground truth range: [{consistent_surface_pressures.min():.1f}, {consistent_surface_pressures.max():.1f}] Pa")
    
    # Heuristic check if predictions are normalized
    pred_range = predictions.max() - predictions.min()
    gt_range = float(consistent_surface_pressures.max() - consistent_surface_pressures.min())
    if pred_range < max(1e-8, gt_range * 0.1):
        # Denormalize using AB-UPT stats
        abupt_pressure_mean = float(stats.surface_pressure_mean[0])
        abupt_pressure_std = float(stats.surface_pressure_std[0])
        logger.info("AB-UPT: Predictions appear normalized, denormalizing...")
        logger.info(f"AB-UPT normalization: mean={abupt_pressure_mean:.4f}, std={abupt_pressure_std:.4f}")
        predictions = predictions * abupt_pressure_std + abupt_pressure_mean
        logger.info(f"AB-UPT: Denormalized prediction range: [{predictions.min():.1f}, {predictions.max():.1f}] Pa")
    else:
        logger.info("AB-UPT: Predictions appear to be in physical units already")
    
    # IMPORTANT: Reorder predictions back to original sampled point order.
    # The collator randomly permutes anchors; we must invert that permutation.
    try:
        # Get normalized positions exactly as in PositionNormalizationPreprocessor
        surface_min = float(stats.surface_raw_pos_min[0])
        surface_max = float(stats.surface_raw_pos_max[0])
        scale = 1000.0
        denom = max(1e-12, (surface_max - surface_min))
        
        def normalize_positions(arr: np.ndarray) -> np.ndarray:
            return ((arr - surface_min) / denom) * scale
        
        # Original sampled positions normalized
        norm_input_positions = normalize_positions(consistent_surface_points.astype(np.float64))
        
        # Anchor positions from batch (already normalized by collator)
        # shape: [1, N, 3] -> [N, 3]
        anchor_positions_norm = batch["surface_anchor_position"].squeeze(0).cpu().numpy().astype(np.float64)
        
        # Map each anchor back to its original index via exact NN match
        tree = cKDTree(norm_input_positions)
        _, mapped_indices = tree.query(anchor_positions_norm, k=1)
        
        if len(mapped_indices) != len(predictions):
            logger.warning("AB-UPT: Mismatch between mapped indices and predictions length; skipping reorder")
        else:
            ordered_predictions = np.empty_like(predictions)
            ordered_predictions[mapped_indices] = predictions
            predictions = ordered_predictions
            logger.info("AB-UPT: Reordered predictions to match input point order")
    except Exception as e:
        logger.warning(f"AB-UPT: Failed to reorder predictions to input order: {e}")
    
    logger.info(f"AB-UPT: Final prediction range: [{predictions.min():.1f}, {predictions.max():.1f}] Pa")
    
    return predictions


def calculate_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics with GLOBAL relative errors for consistency."""
    # Flatten arrays for consistent computation
    pred_flat = predictions.flatten()
    gt_flat = ground_truth.flatten()
    
    # Standard metrics
    mae = np.mean(np.abs(pred_flat - gt_flat))
    mse = np.mean((pred_flat - gt_flat)**2)
    rmse = np.sqrt(mse)
    
    # GLOBAL Relative L2 error (consistent with other models)
    error_l2_norm = np.linalg.norm(pred_flat - gt_flat)
    target_l2_norm = np.linalg.norm(gt_flat)
    rel_l2 = error_l2_norm / (target_l2_norm + 1e-8)
    
    # GLOBAL Relative L1 error
    error_l1_norm = np.sum(np.abs(pred_flat - gt_flat))
    target_l1_norm = np.sum(np.abs(gt_flat))
    rel_l1 = error_l1_norm / (target_l1_norm + 1e-8)
    
    # R-squared
    ss_res = np.sum((gt_flat - pred_flat)**2)
    ss_tot = np.sum((gt_flat - np.mean(gt_flat))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'rel_l2': rel_l2,
        'rel_l1': rel_l1,
        'r2': r2,
        'max_error': np.max(np.abs(pred_flat - gt_flat))
    }


def interpolate_to_full_mesh(mesh_points: np.ndarray, sample_points: np.ndarray, 
                            values: np.ndarray) -> np.ndarray:
    """
    Interpolate predictions from sampled points to full mesh.
    """
    try:
        # Try RBF interpolation first
        interpolator = RBFInterpolator(sample_points, values, 
                                     kernel='linear', epsilon=2)
        return interpolator(mesh_points)
    except:
        # Fallback to nearest neighbor
        tree = cKDTree(sample_points)
        _, indices = tree.query(mesh_points)
        return values[indices]


def save_combined_vtk_files(mesh: pv.PolyData, results: Dict, output_dir: str) -> Tuple[str, str]:
    """
    Save two separate VTK files:
    1. One with all model predictions
    2. One with all error maps
    """
    # Create mesh for predictions
    predictions_mesh = mesh.copy()
    predictions_mesh.point_data['ground_truth'] = mesh.point_data['p'].copy()
    
    # Create mesh for errors
    errors_mesh = mesh.copy()
    errors_mesh.point_data['ground_truth'] = mesh.point_data['p'].copy()
    
    # Add each model's predictions and errors
    for model_name, result_data in results.items():
        pred_mesh = result_data['mesh']
        
        # Add predictions
        predictions_mesh.point_data[f'{model_name}_prediction'] = pred_mesh.point_data['predicted']
        
        # Add absolute errors
        errors_mesh.point_data[f'{model_name}_error'] = np.abs(
            mesh.point_data['p'] - pred_mesh.point_data['predicted']
        )
        
        # Add signed errors (to see over/under prediction)
        errors_mesh.point_data[f'{model_name}_signed_error'] = (
            pred_mesh.point_data['predicted'] - mesh.point_data['p']
        )
    
    # Save predictions VTK
    predictions_path = os.path.join(output_dir, 'all_models_predictions.vtk')
    predictions_mesh.save(predictions_path)
    logger.info(f"✅ Saved all predictions to: {predictions_path}")
    
    # Save errors VTK
    errors_path = os.path.join(output_dir, 'all_models_errors.vtk')
    errors_mesh.save(errors_path)
    logger.info(f"✅ Saved all error maps to: {errors_path}")
    
    # Print summary of what's in each file
    logger.info("\n📊 VTK File Contents:")
    logger.info(f"\nPredictions file ({predictions_path}):")
    logger.info("  - ground_truth: Original pressure values")
    for model_name in results.keys():
        logger.info(f"  - {model_name}_prediction: Predicted pressure values")
    
    logger.info(f"\nErrors file ({errors_path}):")
    logger.info("  - ground_truth: Original pressure values")
    for model_name in results.keys():
        logger.info(f"  - {model_name}_error: Absolute error |prediction - ground_truth|")
        logger.info(f"  - {model_name}_signed_error: Signed error (prediction - ground_truth)")
    
    return predictions_path, errors_path


def create_comparison_figure(mesh: pv.PolyData, results_dict: Dict, output_path: str):
    """Create comparison visualization with error plots."""
    n_models = len(results_dict)
    n_cols = min(4, n_models + 1)
    n_rows = (n_models + 1 + n_cols - 1) // n_cols
    
    plotter = pv.Plotter(
        shape=(n_rows, n_cols), 
        window_size=[n_cols * 600, n_rows * 400],
        off_screen=True
    )
    
    camera_position = [(-5.18, -4.70, 3.44), (1.57, 0.07, 0.20), (0.33, 0.17, 0.93)]
    
    # Ground truth
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars='p', cmap='coolwarm', clim=[-400, 400], show_scalar_bar=True)
    plotter.add_text("Ground Truth", font_size=12, position='upper_edge')
    plotter.camera_position = camera_position
    plotter.camera.zoom(1.1)
    
    # Model errors
    for idx, (model_name, result_data) in enumerate(results_dict.items(), 1):
        pred_mesh = result_data['mesh']
        metrics = result_data['surface_metrics']
        
        row = idx // n_cols
        col = idx % n_cols
        
        plotter.subplot(row, col)
        
        # Compute error
        error = np.abs(mesh.point_data['p'] - pred_mesh.point_data['predicted'])
        error_mesh = mesh.copy()
        error_mesh.point_data['error'] = error
        
        plotter.add_mesh(error_mesh, scalars='error', cmap='Reds', clim=[0, 100], show_scalar_bar=True)
        
        # Add metrics text
        text = f"{model_name}\n"
        text += f"MAE: {metrics['mae']:.1f} Pa\n"
        text += f"RMSE: {metrics['rmse']:.1f} Pa\n"
        text += f"Rel L2: {metrics['rel_l2']:.3f}\n"
        text += f"R²: {metrics['r2']:.3f}"
        
        plotter.add_text(text, font_size=10, position='upper_edge')
        
        plotter.camera_position = camera_position
        plotter.camera.zoom(1.1)
    
    plotter.link_views()
    plotter.screenshot(output_path)
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description='Fixed single car inference with consistent evaluation')
    parser.add_argument('--mesh', type=str, required=True, help='Path to VTK mesh file')
    parser.add_argument('--models', nargs='+', default=None, help='Models to run (default: all)')
    parser.add_argument('--output-dir', type=str, default='inference_results_fixed', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num-points', type=int, default=10000, help='Surface points for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load mesh
    logger.info(f"Loading mesh: {args.mesh}")
    mesh = pv.read(args.mesh)
    logger.info(f"Mesh: {mesh.n_points} points, pressure range: [{mesh.point_data['p'].min():.1f}, {mesh.point_data['p'].max():.1f}] Pa")
    
    # ================================================================
    # CONSISTENT SURFACE POINTS: Same for ALL models
    # ================================================================
    logger.info(f"\n{'='*80}")
    logger.info("🎯 GENERATING CONSISTENT EVALUATION POINTS")
    logger.info(f"{'='*80}")
    
    consistent_surface_points, consistent_surface_pressures = sample_consistent_surface_points(
        mesh, args.num_points, seed=args.seed
    )
    
    # Determine models to run
    if args.models:
        models_to_run = args.models
    else:
        models_to_run = [name for name, path in MODEL_CHECKPOINT_PATHS.items() 
                        if os.path.exists(path)]
    
    # Exclude problematic models
    models_to_exclude = ['gpan', 'pointbert']
    models_to_run = [m for m in models_to_run if m not in models_to_exclude]
    
    logger.info(f"Running {len(models_to_run)} models: {models_to_run}")
    
    # ================================================================
    # RUN INFERENCE WITH CONSISTENT EVALUATION
    # ================================================================
    results = {}
    for model_name in models_to_run:
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 RUNNING {model_name.upper()}")
        logger.info(f"{'='*80}")
        
        try:
            checkpoint_path = MODEL_CHECKPOINT_PATHS[model_name]
            model, wrapper, config = load_wrapper_and_model(model_name, checkpoint_path, device)
            
            if model_name.lower() == 'abupt':
                # ================================================================
                # AB-UPT: Special surface-based inference with anchor/query system
                # ================================================================
                
                # Run AB-UPT inference
                predictions = run_inference_abupt(
                    model, consistent_surface_points, consistent_surface_pressures, mesh, device
                )
                
                surface_predictions = predictions
                surface_ground_truth = consistent_surface_pressures
                
                logger.info(f"✅ AB-UPT EVALUATION: Using all {len(predictions)} predictions (surface points)")
                
            elif 'transolver' in model_name.lower():
                # ================================================================
                # TRANSOLVER: Use interior points for inference, evaluate on consistent surface points
                # ================================================================
                
                # Generate deterministic sample_id based on mesh filename
                mesh_basename = os.path.basename(args.mesh).replace('.vtk', '')
                sample_id = hash(mesh_basename) % 10000
                
                # Prepare Transolver data (consistent surface + interior points)
                transolver_points, transolver_pressures = prepare_transolver_data(
                    consistent_surface_points, consistent_surface_pressures, mesh, model_name, sample_id
                )
                
                # Run Transolver inference
                predictions = run_inference_transolver(
                    model, wrapper, transolver_points, mesh, config,
                    transolver_pressures, device, model_name
                )
                
                # ⚡ CRITICAL: Extract predictions for consistent surface points only
                surface_predictions = predictions[:len(consistent_surface_points)]
                surface_ground_truth = consistent_surface_pressures
                
                logger.info(f"✅ TRANSOLVER EVALUATION: Using first {len(consistent_surface_points)} predictions (surface points)")
                
            else:
                # ================================================================
                # STANDARD MODELS: Use consistent surface points directly
                # ================================================================
                
                predictions = run_inference_standard(
                    model, wrapper, consistent_surface_points, consistent_surface_pressures, device
                )
                
                surface_predictions = predictions
                surface_ground_truth = consistent_surface_pressures
                
                logger.info(f"✅ STANDARD MODEL EVALUATION: Using all {len(predictions)} predictions (surface points)")
            
            # ================================================================
            # CONSISTENT METRICS: Same surface points for all models
            # ================================================================
            
            # Calculate metrics on SAME surface points for fair comparison
            surface_metrics = calculate_metrics(surface_predictions, surface_ground_truth)
            
            # Interpolate to full mesh for visualization
            full_predictions = interpolate_to_full_mesh(
                mesh.points, consistent_surface_points, surface_predictions
            )
            
            # Full mesh metrics (for interpolation quality assessment)
            full_metrics = calculate_metrics(full_predictions, mesh.point_data['p'])
            
            # Create prediction mesh
            pred_mesh = mesh.copy()
            pred_mesh.point_data['predicted'] = full_predictions
            
            # ================================================================
            # DETAILED LOGGING
            # ================================================================
            
            logger.info(f"\n📊 CONSISTENT SURFACE EVALUATION ({len(consistent_surface_points)} points):")
            logger.info(f"  MAE: {surface_metrics['mae']:.1f} Pa")
            logger.info(f"  RMSE: {surface_metrics['rmse']:.1f} Pa")
            logger.info(f"  Rel L2: {surface_metrics['rel_l2']:.3f}")
            logger.info(f"  R²: {surface_metrics['r2']:.3f}")
            
            logger.info(f"\n🌐 FULL MESH INTERPOLATION ({mesh.n_points} points):")
            logger.info(f"  MAE: {full_metrics['mae']:.1f} Pa")
            logger.info(f"  RMSE: {full_metrics['rmse']:.1f} Pa")
            logger.info(f"  Rel L2: {full_metrics['rel_l2']:.3f}")
            logger.info(f"  R²: {full_metrics['r2']:.3f}")
            
            # Store results
            results[model_name] = {
                'mesh': pred_mesh,
                'surface_metrics': surface_metrics,  # Fair comparison metrics
                'full_metrics': full_metrics,        # Interpolation quality
                'surface_predictions': surface_predictions,
                'surface_ground_truth': surface_ground_truth
            }
            
            # Save individual result
            pred_mesh.save(os.path.join(args.output_dir, f"{model_name}_prediction.vtk"))
            
        except Exception as e:
            logger.error(f"❌ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        logger.error("No successful runs!")
        return 1
    
    # ================================================================
    # SAVE COMBINED VTK FILES
    # ================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info("💾 Saving combined VTK files...")
    logger.info(f"{'='*80}")
    
    # Save the two combined VTK files
    predictions_path, errors_path = save_combined_vtk_files(mesh, results, args.output_dir)
    
    # ================================================================
    # CREATE VISUALIZATIONS
    # ================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info("🎨 Creating visualizations...")
    logger.info(f"{'='*80}")
    
    comparison_path = os.path.join(args.output_dir, 'model_comparison.png')
    create_comparison_figure(mesh, results, comparison_path)
    logger.info(f"Comparison saved: {comparison_path}")
    
    # ================================================================
    # SAVE SUMMARY CSV FILES
    # ================================================================
    
    surface_summary_data = []
    full_summary_data = []
    
    for model_name, result_data in results.items():
        surface_metrics = result_data['surface_metrics']
        full_metrics = result_data['full_metrics']
        
        surface_summary_data.append({
            'Model': model_name,
            'MAE (Pa)': surface_metrics['mae'],
            'RMSE (Pa)': surface_metrics['rmse'],
            'Rel L2': surface_metrics['rel_l2'],
            'R²': surface_metrics['r2'],
            'Max Error (Pa)': surface_metrics['max_error']
        })
        
        full_summary_data.append({
            'Model': model_name,
            'MAE (Pa)': full_metrics['mae'],
            'RMSE (Pa)': full_metrics['rmse'],
            'Rel L2': full_metrics['rel_l2'],
            'R²': full_metrics['r2'],
            'Max Error (Pa)': full_metrics['max_error']
        })
    
    surface_df = pd.DataFrame(surface_summary_data)
    surface_df.to_csv(os.path.join(args.output_dir, 'surface_metrics_consistent.csv'), index=False)
    
    full_df = pd.DataFrame(full_summary_data)
    full_df.to_csv(os.path.join(args.output_dir, 'full_mesh_metrics.csv'), index=False)
    
    # ================================================================
    # FINAL COMPARISON TABLES
    # ================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info("🏆 CONSISTENT SURFACE EVALUATION - FAIR COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<20} {'MAE (Pa)':<12} {'RMSE (Pa)':<12} {'Rel L2':<10} {'R²':<10}")
    logger.info("-" * 64)
    
    # Sort by R² for better display
    sorted_surface_data = sorted(surface_summary_data, key=lambda x: x['R²'], reverse=True)
    for data in sorted_surface_data:
        logger.info(f"{data['Model']:<20} {data['MAE (Pa)']:<12.1f} {data['RMSE (Pa)']:<12.1f} "
                   f"{data['Rel L2']:<10.3f} {data['R²']:<10.3f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("🌐 FULL MESH INTERPOLATION QUALITY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<20} {'MAE (Pa)':<12} {'RMSE (Pa)':<12} {'Rel L2':<10} {'R²':<10}")
    logger.info("-" * 64)
    
    sorted_full_data = sorted(full_summary_data, key=lambda x: x['R²'], reverse=True)
    for data in sorted_full_data:
        logger.info(f"{data['Model']:<20} {data['MAE (Pa)']:<12.1f} {data['RMSE (Pa)']:<12.1f} "
                   f"{data['Rel L2']:<10.3f} {data['R²']:<10.3f}")
    
    logger.info("\n🎉 Done! Results saved:")
    logger.info(f"  📁 Predictions: {predictions_path}")
    logger.info(f"  📁 Error maps: {errors_path}")
    logger.info(f"  📊 Comparison: {comparison_path}")
    logger.info(f"  📊 Metrics: surface_metrics_consistent.csv & full_mesh_metrics.csv")
    logger.info("\n💡 Tip: Load the VTK files in ParaView to compare models interactively!")
    
    return 0


if __name__ == "__main__":
    exit(main())