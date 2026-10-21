# models/fno/wrapper.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Wrapper for Fourier Neural Operator (FNO) model
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

# Import the models
from model import FNO_pressure

# Import shared components
regdgcnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'regdgcnn')
sys.path.append(regdgcnn_path)
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from utils import setup_seed

# Import the base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.base_model import BenchmarkModel


class NeuralOperatorWrapper(BenchmarkModel):
    """Wrapper for Fourier Neural Operator (FNO) model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model selection
        self.model_type = config.get('model_type', 'fno')  # 'fno' or 'deeponet'
        
        # Model-specific configuration
        self.model_args = {
            # FNO parameters
            'grid_size': config.get('grid_size', 32),
            'modes': config.get('modes', 8),
            'width': config.get('width', 16),
            
            # General parameters
            'dropout': config.get('dropout', 0.1),
            'num_points': config.get('num_points', 10000)
        }
        
        # Data configuration
        self.dataset_path = config['dataset_path']
        self.subset_dir = config['subset_dir']
        self.num_points = config.get('num_points', 10000)
        self.batch_size = config.get('batch_size', 8)
        self.num_workers = config.get('num_workers', 4)
        self.cache_dir = config.get('cache_dir', None)
        
        # Multi-GPU settings
        self.multi_gpu = config.get('multi_gpu', True)
        self.distributed = config.get('distributed', False)
        
        # Set random seed if specified
        if 'seed' in config:
            setup_seed(config['seed'])
    
    def load_model(self) -> nn.Module:
        """Load the FNO model."""
        model = FNO_pressure(self.model_args)
        print(f"Loading FNO model (grid_size={self.model_args['grid_size']}, modes={self.model_args['modes']})")
        
        # Multi-GPU support
        if self.multi_gpu and torch.cuda.device_count() > 1:
            if not self.distributed:
                model = nn.DataParallel(model)
                print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return model
    
    def get_data_loaders(self) -> Tuple[Any, Any, Any]:
        """Get data loaders."""
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
        """Get loss function."""
        loss_name = self.config.get('loss', 'mse').lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae' or loss_name == 'l1':
            return nn.L1Loss()
        elif loss_name == 'smooth_l1':
            beta = float(self.config.get('huber_delta', 1.0))
            return nn.SmoothL1Loss(beta=beta)
        elif loss_name == 'relative_l2':
            # Custom relative L2 loss for neural operators
            def relative_l2_loss(pred, target):
                return torch.mean(torch.norm(pred - target, dim=1) / (torch.norm(target, dim=1) + 1e-8))
            return relative_l2_loss
        else:
            return nn.MSELoss()
    
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer."""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = float(self.config.get('lr', 0.001))
        weight_decay = float(self.config.get('weight_decay', 1e-4))
        
        params = model.parameters()
        
        if opt_name == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            momentum = float(self.config.get('momentum', 0.9))
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[object]:
        """Get learning rate scheduler."""
        sched_name = self.config.get('scheduler', 'step').lower()
        
        if sched_name == 'reduce_on_plateau':
            return ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        elif sched_name == 'cosine':
            T_max = int(self.config.get('epochs', 100))
            return CosineAnnealingLR(optimizer, T_max=T_max)
        elif sched_name == 'step':
            step_size = int(self.config.get('step_size', 30))
            return StepLR(optimizer, step_size=step_size, gamma=0.7)
        else:
            return None


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
        

        
    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch data."""
        data, targets = batch
        
        # Handle shape requirements
        if data.dim() == 4 and data.shape[1] == 1:
            data = data.squeeze(1)
        if targets.dim() == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        return data, targets
    
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return model(x)
    
    def get_normalization_functions(self) -> Tuple[Callable, Callable]:
        """Get normalization functions."""
        def normalize(x):
            return (x - PRESSURE_MEAN) / PRESSURE_STD
        
        def denormalize(x):
            return x * PRESSURE_STD + PRESSURE_MEAN
        
        return normalize, denormalize
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'architecture': 'Fourier Neural Operator (FNO)',
            'input_points': self.num_points,
            'grid_size': self.model_args['grid_size'],
            'fourier_modes': self.model_args['modes'],
            'width': self.model_args['width']
        })
        
        return info
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            'epochs': self.config.get('epochs', 100),
            'early_stopping_patience': self.config.get('early_stopping_patience', 20),
            'gradient_clip': self.config.get('gradient_clip', 1.0)
        }