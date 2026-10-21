# models/transolver_plus/wrapper.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Wrapper for Transolver++ models with 6D input features [x, y, z, nx, ny, nz] (surface-only mode)
             or optional 7D input features [x, y, z, sdf, nx, ny, nz] (with SDF mode)
FIXED: Changed default to 6D surface-only mode to match regular Transolver
"""

import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np

# Add the model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the models
from model import TransolverPlusPressure, LightweightTransolverPlusPressure

# Use local data loader (with SDF and normals)
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD

# Import the base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.base_model import BenchmarkModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TransolverPlusWrapper(BenchmarkModel):
    """
    Wrapper for Transolver++ models.
    Default: 6D input features [x,y,z,nx,ny,nz] for surface-only mode
    Optional: 7D input features [x,y,z,sdf,nx,ny,nz] when use_sdf=True
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Canonical model identifier - used for all save paths and directories
        self.model_type = 'transolver_plus'
        # Ensure consistent naming across all saving operations
        self.experiment_name = config.get('experiment_name', 'transolver_plus')
        self.name = 'transolver_plus'  # For backward compatibility
        self.model_name = 'transolver_plus'  # For backward compatibility
        
        self.use_lightweight = config.get('use_lightweight', False)
        self.architecture = 'Transolver++'
        
        # Dataset parameters (need to read before model_args)
        self.dataset_path = config['dataset_path']
        self.subset_dir = config['subset_dir']
        self.num_points = config.get('num_points', 10000)
        self.batch_size = config.get('batch_size', 8)
        self.num_workers = config.get('num_workers', 4)
        self.cache_dir = config.get('cache_dir', None)
        self.use_sdf = config.get('use_sdf', False)  # Default: surface-only (6D features [x,y,z,nx,ny,nz])
        
        # Model architecture parameters (include use_sdf for model initialization)
        self.model_args = {
            'n_hidden': config.get('n_hidden', 256),
            'n_layers': config.get('n_layers', 5),
            'n_head': config.get('n_head', 8),
            'mlp_ratio': config.get('mlp_ratio', 1),
            'slice_num': config.get('slice_num', 32),
            'dropout': config.get('dropout', 0.0),
            'unified_pos': config.get('unified_pos', False),
            'ref': config.get('ref', 8),
            'use_sdf': self.use_sdf,  # Pass use_sdf to model
        }
        
        # Training parameters
        self.lr = float(config.get('lr', 0.001))
        self.weight_decay = float(config.get('weight_decay', 0.0))
        
        # Set random seed if provided
        if 'seed' in config:
            setup_seed(config['seed'])
        
        # Store normalization values
        self.pressure_mean = PRESSURE_MEAN
        self.pressure_std = PRESSURE_STD
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")

    def load_model(self) -> nn.Module:
        """Load the Transolver++ model"""
        try:
            if self.use_lightweight:
                model = LightweightTransolverPlusPressure(self.model_args)
                model_name = "Lightweight Transolver++"
            else:
                model = TransolverPlusPressure(self.model_args)
                model_name = "Transolver++"
            
            feature_info = "7D [x,y,z,sdf,nx,ny,nz] with interior points" if self.use_sdf else "6D [x,y,z,nx,ny,nz] surface-only"
            logger.info(f"Loading {model_name} with {feature_info}")
            logger.info(f"Architecture: layers={self.model_args['n_layers']}, "
                       f"hidden={self.model_args['n_hidden']}, heads={self.model_args['n_head']}")
            logger.info(f"Unified position encoding: {self.model_args['unified_pos']}")
            
            # Handle multi-GPU if available
            if torch.cuda.device_count() > 1 and not self.config.get('distributed', False):
                logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
                model = nn.DataParallel(model)
            
            model = model.to(self.device)
            
            # Log model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Physics Attention with {self.model_args['slice_num']} slices")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_data_loaders(self, mode='train') -> Tuple[Any, Any, Any]:
        """Get train, validation, and test data loaders"""
        try:
            world_size = self.config.get('world_size', 1)
            rank = self.config.get('rank', 0)
            
            return get_dataloaders(
                dataset_path=self.dataset_path,
                subset_dir=self.subset_dir,
                num_points=self.num_points,
                batch_size=self.batch_size,
                world_size=world_size,
                rank=rank,
                cache_dir=self.cache_dir,
                num_workers=self.num_workers,
                mode=mode,
                use_sdf=self.use_sdf
            )
        except Exception as e:
            logger.error(f"Error creating dataloaders: {e}")
            raise

    def get_loss_function(self) -> nn.Module:
        """Get loss function for training"""
        return nn.MSELoss()

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer for training"""
        return optim.Adam(
            model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[object]:
        """Get learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'onecycle')
        epochs = self.config.get('epochs', 200)
        
        if scheduler_type == 'onecycle':
            # Estimate steps per epoch
            train_samples = self.config.get('train_samples', 664)  # Update based on actual dataset
            steps_per_epoch = max(1, train_samples // self.batch_size)
            total_steps = epochs * steps_per_epoch
            
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=self.lr * 0.01
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get('step_size', 50),
                gamma=self.config.get('gamma', 0.5)
            )
        else:
            logger.info("No scheduler specified")
            return None

    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess batch data - handle feature tensor based on use_sdf flag
        - use_sdf=False: expects [B, N, 6] with [x,y,z,nx,ny,nz] or [B, N, 3] and computes normals
        - use_sdf=True: expects [B, N, 7] with [x,y,z,sdf,nx,ny,nz]
        """
        try:
            # Handle tuple input from data loader
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                features, pressure = batch
            else:
                features, pressure = batch, None

            # Convert to tensor if necessary
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            if pressure is not None and not isinstance(pressure, torch.Tensor):
                pressure = torch.tensor(pressure, dtype=torch.float32)

            # Handle different input shapes
            if features.dim() == 4 and features.shape[1] == 1:
                features = features.squeeze(1)
            elif features.dim() == 2:
                features = features.unsqueeze(0)

            # Move to device
            features = features.to(self.device)
            if pressure is not None:
                pressure = pressure.to(self.device)

            # Compute expected feature dimension
            expected_features = 7 if self.use_sdf else 6

            # If we get 3D input but expect 6D, compute normals
            if features.shape[-1] == 3 and expected_features == 6:
                # Compute normals using local PCA
                points = features.cpu().numpy()
                normals = np.zeros_like(points)
                
                for b in range(points.shape[0]):
                    for i in range(points.shape[1]):
                        # Get k nearest neighbors
                        k = min(20, points.shape[1])
                        dists = np.sum((points[b] - points[b,i]) ** 2, axis=1)
                        idx = np.argpartition(dists, k)[:k]
                        
                        # Local PCA for normal estimation
                        local_points = points[b,idx] - points[b,i]
                        cov = local_points.T @ local_points
                        _, _, vh = np.linalg.svd(cov)
                        normals[b,i] = vh[-1]  # Smallest eigenvector
                
                # Ensure consistent orientation
                normals = torch.from_numpy(normals).float().to(self.device)
                features = torch.cat([features, normals], dim=-1)

            # Handle pressure shape
            if pressure is not None:
                if pressure.dim() == 3 and pressure.shape[-1] == 1:
                    pressure = pressure.squeeze(-1)
                elif pressure.dim() == 3 and pressure.shape[1] == 1:
                    pressure = pressure.squeeze(1)
                elif pressure.dim() == 1:
                    pressure = pressure.unsqueeze(0)

            # Validate feature dimension
            assert features.dim() == 3 and features.shape[-1] == expected_features, \
                f"Features shape: {features.shape}, expected [B, N, {expected_features}] with " + \
                ("[x,y,z,sdf,nx,ny,nz]" if self.use_sdf else "[x,y,z,nx,ny,nz]")

            if pressure is not None:
                assert pressure.dim() == 2, f"Pressure shape: {pressure.shape}, expected [B, N]"
                assert features.shape[:2] == pressure.shape, \
                    f"Shape mismatch: features {features.shape[:2]} vs pressure {pressure.shape}"

            return features, pressure

        except Exception as e:
            logger.error(f"Error preprocessing batch: {str(e)}")
            raise

    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        Input x should be:
        - [B, N, 6] for surface-only mode with features [x,y,z,nx,ny,nz]
        - [B, N, 7] for SDF mode with features [x,y,z,sdf,nx,ny,nz]
        """
        try:
            with torch.cuda.amp.autocast(enabled=self.config.get('use_amp', False)):
                return model(x)
        except Exception as e:
            use_sdf = self.config.get('use_sdf', False)
            expected_features = 7 if use_sdf else 6
            feature_desc = "[x,y,z,sdf,nx,ny,nz]" if use_sdf else "[x,y,z,nx,ny,nz]"
            logger.error(f"Error in forward pass: {e}")
            logger.error(f"Input shape: {x.shape}")
            logger.error(f"Expected input: [B, N, {expected_features}] with features {feature_desc}")
            raise

    def get_normalization_functions(self) -> Tuple[Callable, Callable]:
        """Get normalization and denormalization functions for pressure only"""
        # Data loader returns raw data, so trainer will normalize
        def normalize(x):
            return (x - self.pressure_mean) / self.pressure_std
        
        def denormalize(x):
            return x * self.pressure_std + self.pressure_mean
        
        return normalize, denormalize
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging - override to ensure consistent naming.
        
        Returns:
            Dictionary with model information using 'transolver_plus' as the name
        """
        return {
            'name': 'transolver_plus',  # Force underscore version
            'config': self.config
        }

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics - denormalizes inputs to physical units (Pa) for meaningful metrics"""
        try:
            with torch.no_grad():
                # Denormalize predictions and targets to physical pressure units (Pa)
                # Training uses normalized values, but evaluation metrics should be in physical units
                pred_denorm = predictions * self.pressure_std + self.pressure_mean
                target_denorm = targets * self.pressure_std + self.pressure_mean
                
                pred_flat = pred_denorm.flatten()
                target_flat = target_denorm.flatten()
                
                # All metrics on denormalized (physical) values for meaningful interpretation
                mse = torch.mean((pred_flat - target_flat) ** 2).item()
                mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
                rmse = torch.sqrt(torch.tensor(mse)).item()
                
                # GLOBAL Relative L2 error (consistent with other models)
                error_norm = torch.norm(pred_flat - target_flat)
                target_norm = torch.norm(target_flat)
                rel_l2 = (error_norm / (target_norm)).item()
                
                # Relative L1 error (also global for consistency)
                error_l1 = torch.sum(torch.abs(pred_flat - target_flat))
                target_l1 = torch.sum(torch.abs(target_flat))
                rel_l1 = (error_l1 / (target_l1)).item()
                
                # Maximum error
                max_error = torch.max(torch.abs(pred_flat - target_flat)).item()
                
                # R-squared
                total_variance = torch.var(target_flat, unbiased=False)
                if total_variance > 1e-8:
                    explained_variance = 1 - torch.mean((pred_flat - target_flat) ** 2) / total_variance
                    r2 = explained_variance.item()
                else:
                    r2 = 0.0
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'relative_l2': rel_l2,
                'relative_l1': rel_l1,
                'max_error': max_error,
                'r2': r2
            }
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return model
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Normalize state-dict keys to match the target module. Strip any
            # DataParallel 'module.' prefix and load into the unwrapped module,
            # then align the optional nested 'model.' prefix to whatever the
            # target actually expects (checkpoints exist both with and without it).
            def _strip(prefix, sd):
                return {k[len(prefix):] if k.startswith(prefix) else k: v
                        for k, v in sd.items()}

            state_dict = _strip('module.', state_dict)

            target = model.module if isinstance(model, nn.DataParallel) else model
            target_keys = set(target.state_dict().keys())
            if not (set(state_dict.keys()) & target_keys):
                with_prefix = {k if k.startswith('model.') else f'model.{k}': v
                               for k, v in state_dict.items()}
                without_prefix = _strip('model.', state_dict)
                if set(with_prefix.keys()) & target_keys:
                    state_dict = with_prefix
                elif set(without_prefix.keys()) & target_keys:
                    state_dict = without_prefix

            target.load_state_dict(state_dict)
            logger.info("Checkpoint loaded successfully")
            
            # Log additional checkpoint info if available
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
                if 'best_val_loss' in checkpoint:
                    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise