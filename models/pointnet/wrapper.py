# models/pointnet/wrapper.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Wrapper for PointNet model to fit the benchmark interface.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Dict, Any, Tuple, Optional, Callable

# Add the model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model
from model import PointNet_pressure

# Import shared components from the regdgcnn directory since they contain the data loader
regdgcnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'regdgcnn')
sys.path.append(regdgcnn_path)
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from utils import setup_seed

# Import the base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.base_model import BenchmarkModel


class PointNetWrapper(BenchmarkModel):
    """Wrapper for PointNet model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model-specific configuration
        self.model_args = {
            'feature_transform': config.get('feature_transform', False),
            'dropout': config.get('dropout', 0.3),
            'output_channels': config.get('output_channels', 1)
        }
        
        # Data configuration
        self.dataset_path = config['dataset_path']
        self.subset_dir = config['subset_dir']
        self.num_points = config.get('num_points', 10000)
        self.batch_size = config.get('batch_size', 12)
        self.num_workers = config.get('num_workers', 4)
        self.cache_dir = config.get('cache_dir', None)
        
        # Multi-GPU settings
        self.multi_gpu = config.get('multi_gpu', True)
        self.distributed = config.get('distributed', False)
        self.gpus = config.get('gpus', None)
        
        # Set random seed if specified
        if 'seed' in config:
            setup_seed(config['seed'])
    
    def load_model(self) -> nn.Module:
        """Load the PointNet model with multi-GPU support."""
        model = PointNet_pressure(self.model_args)
        
        # Multi-GPU support
        if self.multi_gpu and torch.cuda.device_count() > 1:
            if self.distributed:
                # For Distributed Data Parallel (DDP)
                model = model.to(self.device)
            else:
                # For Data Parallel
                available_gpus = list(range(torch.cuda.device_count()))
                print(f"Using GPUs: {available_gpus} (mapped from CUDA_VISIBLE_DEVICES)")
                model = nn.DataParallel(model)
                model = model.to(self.device)
        else:
            model = model.to(self.device)
        
        return model
    
    def get_data_loaders(self) -> Tuple[Any, Any, Any]:
        """Get data loaders with support for distributed training."""
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
            num_workers=self.num_workers
        )
    
    def get_loss_function(self) -> nn.Module:
        """Get loss function based on configuration."""
        loss_name = self.config.get('loss', 'mse').lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae' or loss_name == 'l1':
            return nn.L1Loss()
        elif loss_name == 'smooth_l1' or loss_name == 'huber':
            beta = float(self.config.get('huber_delta', 1.0))
            return nn.SmoothL1Loss(beta=beta)
        elif loss_name == 'mse_mae':
            # Combined loss
            mse_weight = float(self.config.get('mse_weight', 0.5))
            mae_weight = float(self.config.get('mae_weight', 0.5))
            return lambda pred, target: (
                mse_weight * nn.MSELoss()(pred, target) + 
                mae_weight * nn.L1Loss()(pred, target)
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = float(self.config.get('lr', 0.001))
        weight_decay = float(self.config.get('weight_decay', 1e-4))
        
        # Get parameters - handle both DataParallel and regular models
        if isinstance(model, nn.DataParallel):
            params = model.module.parameters()
        else:
            params = model.parameters()
        
        if opt_name == 'adam':
            beta1 = float(self.config.get('beta1', 0.9))
            beta2 = float(self.config.get('beta2', 0.999))
            eps = float(self.config.get('eps', 1e-8))
            return optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
            
        elif opt_name == 'adamw':
            beta1 = float(self.config.get('beta1', 0.9))
            beta2 = float(self.config.get('beta2', 0.999))
            eps = float(self.config.get('eps', 1e-8))
            return optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
            
        elif opt_name == 'sgd':
            momentum = float(self.config.get('momentum', 0.9))
            nesterov = self.config.get('nesterov', True)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
            
        else:
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[object]:
        """Get learning rate scheduler based on configuration."""
        sched_name = self.config.get('scheduler', 'reduce_on_plateau').lower()
        
        if sched_name == 'reduce_on_plateau' or sched_name == 'plateau':
            mode = self.config.get('scheduler_mode', 'min')
            patience = int(self.config.get('scheduler_patience', 10))
            factor = float(self.config.get('scheduler_factor', 0.1))
            min_lr = float(self.config.get('min_lr', 1e-7))
            return ReduceLROnPlateau(
                optimizer, mode=mode, patience=patience, 
                factor=factor, min_lr=min_lr
            )
            
        elif sched_name == 'cosine' or sched_name == 'cosine_annealing':
            T_max = int(self.config.get('T_max', self.config.get('epochs', 150)))
            eta_min = float(self.config.get('min_lr', 1e-6))
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            
        elif sched_name == 'step':
            step_size = int(self.config.get('step_size', 30))
            gamma = float(self.config.get('gamma', 0.1))
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
            
        elif sched_name == 'none' or sched_name is None:
            return None
            
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")
    
    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch data for PointNet."""
        data, targets = batch
        
        # Handle the specific shape requirements
        # data should be (batch_size, 3, num_points)
        # targets should be (batch_size, num_points)
        if data.dim() == 4 and data.shape[1] == 1:
            data = data.squeeze(1)
        if targets.dim() == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        return data, targets
    
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        output = model(x)
        
        # Output should already be (batch_size, num_points) from the fixed model
        return output
    
    def get_normalization_functions(self) -> Tuple[Callable, Callable]:
        """Get normalization functions for pressure values."""
        def normalize(x):
            return (x - PRESSURE_MEAN) / PRESSURE_STD
        
        def denormalize(x):
            return x * PRESSURE_STD + PRESSURE_MEAN
        
        return normalize, denormalize
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        info.update({
            'architecture': 'PointNet',
            'input_points': self.num_points,
            'feature_transform': self.model_args['feature_transform'],
            'dropout': self.model_args['dropout'],
            'optimizer': self.config.get('optimizer', 'adam'),
            'learning_rate': self.config.get('lr', 0.001),
            'scheduler': self.config.get('scheduler', 'reduce_on_plateau'),
            'loss_function': self.config.get('loss', 'mse'),
            'multi_gpu': self.multi_gpu,
            'distributed': self.distributed
        })
        return info
    

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """Load model weights from checkpoint."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract state dict - handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapper - remove 'module.' prefix if present
            is_dataparallel_checkpoint = any(k.startswith('module.') for k in state_dict.keys())
            is_dataparallel_model = isinstance(model, nn.DataParallel)
            
            if is_dataparallel_checkpoint and not is_dataparallel_model:
                # Checkpoint from DataParallel, loading to single model
                logger.info("Removing 'module.' prefix from checkpoint (DataParallel -> single model)")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif not is_dataparallel_checkpoint and is_dataparallel_model:
                # Checkpoint from single model, loading to DataParallel
                logger.info("Adding 'module.' prefix to checkpoint (single model -> DataParallel)")
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
            # Load the state dict
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("Checkpoint loaded successfully")
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}")
                logger.info("Trying with strict=False")
                incompatible = model.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    logger.warning(f"Missing keys: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            
            # Log additional checkpoint info if available
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
                if 'best_val_loss' in checkpoint:
                    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")


    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            'epochs': self.config.get('epochs', 150),
            'early_stopping_patience': self.config.get('early_stopping_patience', 20),
            'gradient_clip': self.config.get('gradient_clip', 1.0)
        }