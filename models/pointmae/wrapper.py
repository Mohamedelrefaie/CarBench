# models/pointmae/wrapper.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Wrapper for Point-MAE models
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Tuple, Optional, Callable

# Add the model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the models
from model import PointMAEForPressure, PointMAELightweight

# Import shared components
regdgcnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'regdgcnn')
sys.path.append(regdgcnn_path)
try:
    from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
except ImportError:
    PRESSURE_MEAN = -37.0900
    PRESSURE_STD = 48.0955
    get_dataloaders = None

try:
    from utils import setup_seed
except ImportError:
    import random
    import numpy as np
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

# Import the base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.base_model import BenchmarkModel


class PointMAEWrapper(BenchmarkModel):
    """Wrapper for Point-MAE models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model selection
        self.use_lightweight = config.get('use_lightweight', False)
        self.architecture = config.get('architecture', 'pointmae')  # Default architecture
        
        # Model-specific configuration
        self.model_config = {
            'num_points': config.get('num_points', 10000),
            'embed_dim': config.get('embed_dim', 768),
            'depth': config.get('depth', 12),
            'num_heads': config.get('num_heads', 12),
            'mlp_ratio': config.get('mlp_ratio', 4.0),
            'num_group': config.get('num_group', 128),
            'group_size': config.get('group_size', 32),
            'drop_rate': config.get('drop_rate', 0.0),
            'attn_drop_rate': config.get('attn_drop_rate', 0.0),
            'drop_path_rate': config.get('drop_path_rate', 0.1),
            'use_pretrained': config.get('use_pretrained', True),
            'pretrained_path': config.get('pretrained_path', None),
            'architecture': self.architecture
        }
        
        # Data configuration
        self.dataset_path = config['dataset_path']
        self.subset_dir = config['subset_dir']
        self.num_points = config.get('num_points', 10000)
        self.batch_size = config.get('batch_size', 24)
        self.num_workers = config.get('num_workers', 4)
        self.cache_dir = config.get('cache_dir', None)
        
        # Multi-GPU settings
        self.multi_gpu = config.get('multi_gpu', True)
        self.distributed = config.get('distributed', False)
        
        # Set random seed if specified
        if 'seed' in config:
            setup_seed(config['seed'])
    
    def detect_architecture_from_checkpoint(self, checkpoint_path: str) -> str:
        """
        Detect the architecture used in a checkpoint by examining its state_dict keys.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Architecture string ('hybrid', 'efficient', 'pointnet++', 'pointmae')
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}, using config architecture: {self.architecture}")
            return self.architecture
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'module.' prefix if present
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v
            
            # Check architecture patterns in state_dict keys
            keys = list(clean_state_dict.keys())
            
            if any('model.encoder.0.weight' in k for k in keys):
                detected_arch = 'efficient'
            elif any('model.local_feat.edge1.conv.0.weight' in k for k in keys):
                detected_arch = 'hybrid'
            elif any('model.sa1.0.weight' in k for k in keys):
                detected_arch = 'pointnet++'
            elif any('model.conv1.weight' in k for k in keys):
                detected_arch = 'pointmae'
            else:
                print("Warning: Could not detect architecture from checkpoint keys")
                print(f"Sample keys: {keys[:10]}")
                detected_arch = self.architecture  # Use config default
            
            print(f"Detected architecture from checkpoint: {detected_arch}")
            return detected_arch
            
        except Exception as e:
            print(f"Error reading checkpoint for architecture detection: {e}")
            print(f"Using config architecture: {self.architecture}")
            return self.architecture
    
    def load_model(self) -> nn.Module:
        """Load the Point-MAE model."""
        if self.use_lightweight:
            print("Loading Lightweight Point-MAE...")
            model = PointMAELightweight(num_points=self.num_points)
        else:
            arch_name = self.architecture
            if arch_name == "hybrid":
                model_type = "Hybrid Transformer-DGCNN"
            elif arch_name == "pointnet++":
                model_type = "PointNet++ Regression"
            elif arch_name == "efficient":
                model_type = "Efficient PressureNet"
            else:
                model_type = "Point-MAE"
                
            print(f"Loading {model_type} (architecture={arch_name}, "
                  f"depth={self.model_config['depth']}, "
                  f"num_groups={self.model_config['num_group']})")
            
            # Create model with architecture parameter
            model = PointMAEForPressure(**self.model_config)
        
        # Multi-GPU support
        if self.multi_gpu and torch.cuda.device_count() > 1 and not self.distributed:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Only show frozen params for original Point-MAE with pretrained weights
        if (self.architecture == 'pointmae' and 
            self.model_config['use_pretrained'] and 
            self.model_config['pretrained_path']):
            frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        
        return model
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """
        Load a checkpoint with automatic architecture detection and correction.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: The model to load the checkpoint into
            
        Returns:
            The model with loaded weights
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Detect architecture from checkpoint
        detected_arch = self.detect_architecture_from_checkpoint(checkpoint_path)
        
        # If architectures don't match, recreate the model
        if detected_arch != self.architecture:
            print(f"Architecture mismatch!")
            print(f"  Config architecture: {self.architecture}")
            print(f"  Checkpoint architecture: {detected_arch}")
            print(f"Recreating model with checkpoint architecture: {detected_arch}")
            
            # Update architecture and model config
            self.architecture = detected_arch
            self.model_config['architecture'] = detected_arch
            
            # Recreate model with correct architecture
            model = self.load_model()
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
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
            matched_keys = []
            mismatched_keys = []
            
            for k, v in new_state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        model_dict[k] = v
                        matched_keys.append(k)
                    else:
                        mismatched_keys.append(f"{k}: checkpoint{v.shape} vs model{model_dict[k].shape}")
                else:
                    mismatched_keys.append(f"{k}: not found in model")
            
            model.load_state_dict(model_dict)
            print(f"✓ Partial loading completed: {len(matched_keys)}/{len(new_state_dict)} parameters loaded")
            
            if mismatched_keys:
                print(f"Mismatched keys ({len(mismatched_keys)}):")
                for key in mismatched_keys[:10]:  # Show first 10
                    print(f"  {key}")
                if len(mismatched_keys) > 10:
                    print(f"  ... and {len(mismatched_keys) - 10} more")
        
        return model
        
    
    def get_data_loaders(self) -> Tuple[Any, Any, Any]:
        """Get data loaders."""
        if get_dataloaders is None:
            raise ImportError("Cannot import data loaders. Please ensure regdgcnn is available.")
        
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
        elif loss_name == 'smooth_l1' or loss_name == 'huber':
            beta = float(self.config.get('huber_delta', 1.0))
            return nn.SmoothL1Loss(beta=beta)
        elif loss_name == 'relative_l2':
            def relative_l2_loss(pred, target):
                return torch.mean(torch.norm(pred - target, dim=1) / (torch.norm(target, dim=1) + 1e-8))
            return relative_l2_loss
        else:
            return nn.MSELoss()
    
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer with different learning rates for pretrained and new layers."""
        opt_name = self.config.get('optimizer', 'adamw').lower()
        lr = float(self.config.get('lr', 5e-4))
        weight_decay = float(self.config.get('weight_decay', 0.05))
        
        # Only use different learning rates for original Point-MAE with pretrained weights
        if (self.architecture == 'pointmae' and 
            self.model_config['use_pretrained'] and 
            self.model_config['pretrained_path']):
            # Lower learning rate for pretrained layers
            pretrained_params = []
            new_params = []
            
            for name, param in model.named_parameters():
                if 'pressure_head' in name or 'upsample' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
            
            param_groups = [
                {'params': pretrained_params, 'lr': lr * 0.1},  # 10x smaller for pretrained
                {'params': new_params, 'lr': lr}
            ]
            
            print(f"Using different learning rates: pretrained={lr*0.1}, new={lr}")
        else:
            param_groups = model.parameters()
        
        if opt_name == 'adam':
            return optim.Adam(param_groups, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            momentum = float(self.config.get('momentum', 0.9))
            return optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
        else:
            return optim.AdamW(param_groups, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[object]:
        """Get learning rate scheduler."""
        sched_name = self.config.get('scheduler', 'cosine').lower()
        
        if sched_name == 'cosine':
            T_max = int(self.config.get('epochs', 150))
            eta_min = float(self.config.get('eta_min', 1e-6))
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif sched_name == 'reduce':
            patience = int(self.config.get('scheduler_patience', 10))
            factor = float(self.config.get('scheduler_factor', 0.5))
            return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        else:
            return None
    
    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch data with normalization."""
        data, targets = batch
        
        # Handle shape requirements
        if data.dim() == 4 and data.shape[1] == 1:
            data = data.squeeze(1)
        if targets.dim() == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        # Ensure contiguous
        data = data.contiguous()
        targets = targets.contiguous()
        return data, targets
    
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            model: The Point-MAE model
            x: Point cloud coordinates (B, 3, N) or (B, N, 3)
        """
        return model(x)
    
    def get_normalization_functions(self) -> Tuple[Callable, Callable]:
        """Get normalization functions."""
        mean = self.config.get('pressure_mean', PRESSURE_MEAN)
        std = self.config.get('pressure_std', PRESSURE_STD)
        
        def normalize(x):
            return (x - mean) / std
        
        def denormalize(x):
            return x * std + mean
        
        return normalize, denormalize
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        
        if self.use_lightweight:
            arch_desc = 'Lightweight Point-MAE'
        elif self.architecture == 'hybrid':
            arch_desc = 'Hybrid Transformer-DGCNN'
        elif self.architecture == 'pointnet++':
            arch_desc = 'PointNet++ Regression'
        elif self.architecture == 'efficient':
            arch_desc = 'Efficient PressureNet'
        else:
            arch_desc = 'Point-MAE'
            
        info.update({
            'architecture': arch_desc,
            'architecture_type': self.architecture,
            'input_points': self.num_points,
            'embed_dim': self.model_config['embed_dim'],
            'depth': self.model_config['depth'],
            'num_heads': self.model_config['num_heads'],
            'num_groups': self.model_config['num_group'],
            'group_size': self.model_config['group_size'],
            'use_pretrained': self.model_config['use_pretrained'] if self.architecture == 'pointmae' else False
        })
        
        return info
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            'epochs': self.config.get('epochs', 150),
            'early_stopping_patience': self.config.get('early_stopping_patience', 20),
            'gradient_clip': self.config.get('gradient_clip', 1.0),
            'val_interval': self.config.get('val_interval', 5),
            'warmup_epochs': self.config.get('warmup_epochs', 10)
        }