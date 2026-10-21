"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Wrapper for Large Transolver model to fit the benchmark interface.
"""
import sys, os
from typing import Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LargeTransolverPressure
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD

# Add parent directory for data_loader
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from benchmark.base_model import BenchmarkModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LargeTransolverWrapper(BenchmarkModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = 'transolver_large'
        self.architecture = 'LargeTransolver'
        self.dataset_path = config['dataset_path']
        self.subset_dir = config['subset_dir']
        self.num_points = config.get('num_points', 10000)
        self.batch_size = config.get('batch_size', 8)
        self.num_workers = config.get('num_workers', 4)
        self.cache_dir = config.get('cache_dir', None)
        self.use_sdf = config.get('use_sdf', False)
        self.model_args = {
            'n_hidden': config.get('n_hidden', 384),
            'n_layers': config.get('n_layers', 7),
            'n_head': config.get('n_head', 12),
            'mlp_ratio': config.get('mlp_ratio', 2),
            'slice_num': config.get('slice_num', 48),
            'dropout': config.get('dropout', 0.1),
            'unified_pos': config.get('unified_pos', False),
            'ref': config.get('ref', 8),
            'use_sdf': self.use_sdf,
        }
        self.lr = float(config.get('lr', 0.001))
        self.weight_decay = float(config.get('weight_decay', 0.0))
        if 'seed' in config:
            setup_seed(config['seed'])
        self.pressure_mean = PRESSURE_MEAN
        self.pressure_std = PRESSURE_STD
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Override to return correct model name for saving checkpoints.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.model_type,  # Use 'transolver_large' instead of class name
            'architecture': self.architecture,
            'config': self.config
        }

    def load_model(self) -> nn.Module:
        """Load and initialize the Large Transolver model"""
        try:
            model = LargeTransolverPressure(self.model_args)
            model_name = "Large Transolver"
            
            feature_info = "7D [x,y,z,sdf,nx,ny,nz] with interior points" if self.use_sdf else "6D [x,y,z,nx,ny,nz] surface+normals"
            logger.info(f"Loading {model_name} with {feature_info}")
            logger.info(f"Architecture: layers={self.model_args['n_layers']}, "
                       f"hidden={self.model_args['n_hidden']}, heads={self.model_args['n_head']}")
            logger.info(f"Unified position encoding: {self.model_args['unified_pos']}")
            
            # Move to device and setup DataParallel if multiple GPUs
            if torch.cuda.device_count() > 1 and not self.config.get('distributed', False):
                logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
                model = nn.DataParallel(model)
            
            model = model.to(self.device)
            
            # Log model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
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
        scheduler_type = self.config.get('scheduler', 'steplr')
        epochs = self.config.get('epochs', 1000)
        
        if scheduler_type == 'onecycle':
            train_samples = self.config.get('train_samples', 5819)
            steps_per_epoch = max(1, train_samples // self.batch_size)
            total_steps = epochs * steps_per_epoch
            
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                div_factor=self.config.get('div_factor', 25.0),
                final_div_factor=self.config.get('final_div_factor', 10000.0),
                pct_start=self.config.get('pct_start', 0.1)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_type.lower() == 'steplr':
            step_size = self.config.get('step_size', 150)
            gamma = self.config.get('gamma', 0.1)
            logger.info(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type.lower() == 'reducelronplateau':
            mode = self.config.get('plateau_mode', 'min')
            factor = self.config.get('plateau_factor', 0.5)
            patience = self.config.get('plateau_patience', 10)
            min_lr = self.config.get('plateau_min_lr', 1e-6)
            logger.info(f"Using ReduceLROnPlateau scheduler: mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr}")
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True
            )
        else:
            logger.warning(f"Unknown scheduler type '{scheduler_type}', training without scheduler")
            return None
        
    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch data"""
        try:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                features, pressure = batch
                
                features = features.to(self.device)
                pressure = pressure.to(self.device)
                
                if features.dim() == 4 and features.shape[1] == 1:
                    features = features.squeeze(1)
                elif features.dim() == 2:
                    features = features.unsqueeze(0)
                
                if pressure.dim() == 3 and pressure.shape[-1] == 1:
                    pressure = pressure.squeeze(-1)
                elif pressure.dim() == 3 and pressure.shape[1] == 1:
                    pressure = pressure.squeeze(1)
                elif pressure.dim() == 1:
                    pressure = pressure.unsqueeze(0)
                
                use_sdf = self.config.get('use_sdf', False)
                expected_features = 7 if use_sdf else 6
                feature_desc = "[x,y,z,sdf,nx,ny,nz]" if use_sdf else "[x,y,z,nx,ny,nz]"
                
                assert features.dim() == 3 and features.shape[-1] == expected_features, \
                    f"Features shape: {features.shape}, expected [B, N, {expected_features}] with {feature_desc}"
                assert pressure.dim() == 2, f"Pressure shape: {pressure.shape}, expected [B, N]"
                assert features.shape[:2] == pressure.shape, \
                    f"Shape mismatch: features {features.shape[:2]} vs pressure {pressure.shape}"
                
                return features, pressure
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            raise
    
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
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
        def normalize(x):
            return (x - self.pressure_mean) / self.pressure_std
        
        def denormalize(x):
            return x * self.pressure_std + self.pressure_mean
        
        return normalize, denormalize

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics"""
        try:
            with torch.no_grad():
                pred_denorm = predictions * self.pressure_std + self.pressure_mean
                target_denorm = targets * self.pressure_std + self.pressure_mean
                
                pred_flat = pred_denorm.flatten()
                target_flat = target_denorm.flatten()
                
                mse = torch.mean((pred_flat - target_flat) ** 2).item()
                mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
                rmse = torch.sqrt(torch.tensor(mse)).item()
                
                error_norm = torch.norm(pred_flat - target_flat)
                target_norm = torch.norm(target_flat)
                rel_l2 = (error_norm / target_norm).item()
                
                error_l1 = torch.sum(torch.abs(pred_flat - target_flat))
                target_l1 = torch.sum(torch.abs(target_flat))
                rel_l1 = (error_l1 / target_l1).item()
                
                max_error = torch.max(torch.abs(pred_flat - target_flat)).item()
                
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
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            # Handle DataParallel prefix mismatch
            model_has_module_prefix = hasattr(model, 'module')
            checkpoint_has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
            
            if model_has_module_prefix and not checkpoint_has_module_prefix:
                # Model wrapped with DataParallel but checkpoint doesn't have module prefix
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif not model_has_module_prefix and checkpoint_has_module_prefix:
                # Checkpoint has module prefix but model is not wrapped with DataParallel
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Remove any 'model.' prefix if present
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            logger.info("Checkpoint loaded successfully")
            
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                if 'best_val_loss' in checkpoint:
                    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    # ...rest of the methods can be copied from the original wrapper if needed...
