# models/pointnet_large/wrapper.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Wrapper for Large PointNet models to fit the benchmark interface.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, OneCycleLR
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Add the model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the models
from model import PointNetLarge_pressure, PointNetXL_pressure

# Import shared components from the regdgcnn directory
regdgcnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'regdgcnn')
sys.path.append(regdgcnn_path)
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from utils import setup_seed

# Import the base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.base_model import BenchmarkModel


class PointNetLargeWrapper(BenchmarkModel):
    """Wrapper for Large PointNet models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model variant selection
        self.model_variant = config.get('model_variant', 'large')  # 'large' or 'xl'
        
        # Model-specific configuration
        self.model_args = {
            'feature_transform': config.get('feature_transform', False),
            'dropout': config.get('dropout', 0.3),
            'dropout_heavy': config.get('dropout_heavy', 0.5),
            'output_channels': config.get('output_channels', 1)
        }
        
        # Add channel configurations if specified
        if 'feat_channels' in config:
            self.model_args['feat_channels'] = config['feat_channels']
        if 'seg_channels' in config:
            self.model_args['seg_channels'] = config['seg_channels']
        
        # Data configuration
        self.dataset_path = config['dataset_path']
        self.subset_dir = config['subset_dir']
        self.num_points = config.get('num_points', 10000)
        self.batch_size = config.get('batch_size', 8)  # Smaller batch size for larger model
        self.num_workers = config.get('num_workers', 4)
        self.cache_dir = config.get('cache_dir', None)
        
        # Multi-GPU settings
        self.multi_gpu = config.get('multi_gpu', True)
        self.distributed = config.get('distributed', False)
        self.gpus = config.get('gpus', None)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        
        # Set random seed if specified
        if 'seed' in config:
            setup_seed(config['seed'])
    
    def load_model(self) -> nn.Module:
        """Load the Large PointNet model with multi-GPU support."""
        if self.model_variant == 'xl':
            model = PointNetXL_pressure(self.model_args)
            print("Loading PointNet-XL model")
        else:
            model = PointNetLarge_pressure(self.model_args)
            print("Loading PointNet-Large model")
        
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
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model


    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """
        Load model checkpoint with proper handling of DataParallel.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: The model to load the checkpoint into
            
        Returns:
            The model with loaded weights
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return model
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Check if model is wrapped in DataParallel
        model_is_dp = isinstance(model, nn.DataParallel)
        state_has_module = any(k.startswith('module.') for k in state_dict.keys())
        
        # Fix state dict keys based on model structure
        new_state_dict = {}
        if state_has_module and not model_is_dp:
            # Remove 'module.' prefix
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
        elif not state_has_module and model_is_dp:
            # Add 'module.' prefix
            for k, v in state_dict.items():
                new_state_dict[f'module.{k}'] = v
        else:
            # Keys already match
            new_state_dict = state_dict
        
        # Load state dict
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("✓ Model checkpoint loaded successfully!")
        except Exception as e:
            print(f"Strict loading failed: {e}")
            print("Attempting partial loading...")
            
            # Try partial loading
            model_dict = model.state_dict()
            matched = 0
            for k, v in new_state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    matched += 1
            
            model.load_state_dict(model_dict)
            print(f"✓ Loaded {matched}/{len(new_state_dict)} parameters")
        
        # Print checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
        
        return model


    def pointnet_collate_fn(self, batch):
        """
        Custom collate function to handle variable-sized point clouds.
        Ensures all samples have exactly num_points by sampling or padding.
        """
        points_list, targets_list = zip(*batch)
        
        # Process each sample to ensure consistent point count
        processed_points = []
        processed_targets = []
        
        for points, targets in zip(points_list, targets_list):
            # points shape: [1, 3, N] where N varies
            # targets shape: [1, M] where M varies
            
            if points is None or targets is None:
                logger.warning("Skipping None sample in batch")
                continue
                
            # Remove batch dimension for processing
            if points.dim() == 3 and points.shape[0] == 1:
                points = points.squeeze(0)  # [3, N]
            if targets.dim() == 2 and targets.shape[0] == 1:
                targets = targets.squeeze(0)  # [M]
                
            current_num_points = points.shape[1]
            target_num_points = self.num_points
            
            # Ensure we have exactly target_num_points
            if current_num_points > target_num_points:
                # Randomly sample target_num_points
                indices = torch.randperm(current_num_points)[:target_num_points]
                points = points[:, indices]  # [3, target_num_points]
                targets = targets[indices]   # [target_num_points]
                
            elif current_num_points < target_num_points:
                # Pad by repeating random points
                deficit = target_num_points - current_num_points
                repeat_indices = torch.randint(0, current_num_points, (deficit,))
                
                # Concatenate original with repeated points
                repeated_points = points[:, repeat_indices]
                repeated_targets = targets[repeat_indices]
                
                points = torch.cat([points, repeated_points], dim=1)
                targets = torch.cat([targets, repeated_targets], dim=0)
            
            # Verify final shapes
            assert points.shape == (3, target_num_points), f"Points shape: {points.shape}, expected: (3, {target_num_points})"
            assert targets.shape == (target_num_points,), f"Targets shape: {targets.shape}, expected: ({target_num_points},)"
            
            # Add batch dimension back
            processed_points.append(points.unsqueeze(0))  # [1, 3, target_num_points]
            processed_targets.append(targets.unsqueeze(0))  # [1, target_num_points]
        
        if not processed_points:
            logger.error("No valid samples in batch after processing")
            return None, None
            
        # Stack into batch tensors
        batch_points = torch.cat(processed_points, dim=0)   # [B, 3, target_num_points]
        batch_targets = torch.cat(processed_targets, dim=0) # [B, target_num_points]
        
        return batch_points, batch_targets

    def get_data_loaders(self) -> Tuple[Any, Any, Any]:
        """Get data loaders with support for distributed training and custom collate function."""
        world_size = self.config.get('world_size', 1)
        rank = self.config.get('rank', 0)
        
        # Get the standard data loaders first
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_path=self.dataset_path,
            subset_dir=self.subset_dir,
            num_points=self.num_points,
            batch_size=self.batch_size,
            world_size=world_size,
            rank=rank,
            cache_dir=self.cache_dir,
            num_workers=self.num_workers
        )
        
        # Create new data loaders with custom collate function to handle variable sizes
        train_loader_fixed = DataLoader(
            train_loader.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.pointnet_collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader_fixed = DataLoader(
            val_loader.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.pointnet_collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader_fixed = DataLoader(
            test_loader.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.pointnet_collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        logger.info(f"✓ Created PointNet data loaders with fixed {self.num_points} points per sample")
        return train_loader_fixed, val_loader_fixed, test_loader_fixed
    
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
        elif loss_name == 'adaptive':
            # Adaptive loss that focuses more on high-error regions
            def adaptive_loss(pred, target):
                errors = torch.abs(pred - target)
                weights = 1 + errors / (errors.mean() + 1e-6)
                weighted_mse = torch.mean(weights * (pred - target) ** 2)
                return weighted_mse
            return adaptive_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        opt_name = self.config.get('optimizer', 'adamw').lower()
        lr = float(self.config.get('lr', 0.0001))  # Lower LR for large model
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
            
        elif opt_name == 'lamb':
            # LAMB optimizer for large batch training
            try:
                from apex.optimizers import FusedLAMB
                return FusedLAMB(params, lr=lr, weight_decay=weight_decay)
            except ImportError:
                print("LAMB optimizer not available, falling back to AdamW")
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
                
        else:
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[object]:
        """Get learning rate scheduler based on configuration."""
        sched_name = self.config.get('scheduler', 'onecycle').lower()
        
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
            
        elif sched_name == 'onecycle':
            # OneCycle scheduler for super-convergence
            max_lr = float(self.config.get('max_lr', 0.01))
            epochs = int(self.config.get('epochs', 150))
            steps_per_epoch = self.config.get('steps_per_epoch', 100)  # Will be set by trainer
            return OneCycleLR(
                optimizer, 
                max_lr=max_lr,
                total_steps=epochs * steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            
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
        if batch[0] is None or batch[1] is None:
            logger.error("Received None batch from collate function")
            raise ValueError("Invalid batch data")
            
        data, targets = batch
        
        # Debug logging for batch shapes
        logger.debug(f"Batch shapes - Data: {data.shape}, Targets: {targets.shape}")
        
        # Verify consistent shapes within batch
        expected_points = self.num_points
        if data.shape[-1] != expected_points:
            logger.warning(f"Unexpected data shape: {data.shape}, expected last dim: {expected_points}")
        if targets.shape[-1] != expected_points:
            logger.warning(f"Unexpected targets shape: {targets.shape}, expected last dim: {expected_points}")
        
        # Handle the specific shape requirements
        if data.dim() == 4 and data.shape[1] == 1:
            data = data.squeeze(1)
        if targets.dim() == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        # Final shape verification
        logger.debug(f"Final batch shapes - Data: {data.shape}, Targets: {targets.shape}")
        
        return data, targets
    
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        output = model(x)
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
            'architecture': f'PointNet-{self.model_variant.upper()}',
            'input_points': self.num_points,
            'feature_transform': self.model_args['feature_transform'],
            'dropout': self.model_args['dropout'],
            'dropout_heavy': self.model_args.get('dropout_heavy', 0.5),
            'optimizer': self.config.get('optimizer', 'adamw'),
            'learning_rate': self.config.get('lr', 0.0001),
            'scheduler': self.config.get('scheduler', 'onecycle'),
            'loss_function': self.config.get('loss', 'mse'),
            'multi_gpu': self.multi_gpu,
            'distributed': self.distributed,
            'use_amp': self.use_amp
        })
        return info
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            'epochs': self.config.get('epochs', 150),
            'early_stopping_patience': self.config.get('early_stopping_patience', 20),
            'gradient_clip': self.config.get('gradient_clip', 1.0),
            'use_amp': self.use_amp
        }