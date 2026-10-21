#!/usr/bin/env python
"""
Complete AB-UPT Wrapper for DrivAerNet Benchmark
Fixed version with proper normalization and all dataset splits
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Callable
import logging
from tqdm import tqdm
import time
import sys
import os
import traceback

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import AB-UPT components
from surface_abupt_collator import SurfaceOnlyAbuptCollator
from surface_only_model import SurfaceOnlyABUPT
from drivaernet_dataset import DrivAerNetDataset
import preprocessors

# Import the base class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.base_model import BenchmarkModel


class ABUPTWrapper(BenchmarkModel):
    """Complete wrapper for pretrained AB-UPT model with full BenchmarkModel interface"""
    
    def __init__(self, config: Dict):
        """Initialize the AB-UPT wrapper
        
        Args:
            config: Configuration dictionary containing model and dataset parameters
        """
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model hyperparameters
        self.num_geometry_points = config.get('num_geometry_points', 65536)
        self.num_surface_anchor_points = config.get('num_surface_anchor_points', 16384)
        self.num_geometry_supernodes = config.get('num_geometry_supernodes', 16384)
        self.num_surface_queries = config.get('num_surface_queries', 16384)
        
        # Training/evaluation parameters
        self.batch_size = config.get('batch_size', 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = config.get('num_workers', 4)
        
        # Dataset paths
        self.dataset_path = Path(config['dataset_path'])
        self.subset_dir = Path(config.get('subset_dir', ''))
        
        # Find checkpoint path
        self.checkpoint_path = self._find_checkpoint_path(config)
        
        # Initialize components
        self.model = None
        self.collator = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Normalization parameters from DrivAerNet dataset
        self.pressure_mean = -93.41  # Pa
        self.pressure_std = 120.79   # Pa
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Cache for current batch processing
        self._current_batch = None
        self._current_split = 'test'  # Track current evaluation split
    
    def _find_checkpoint_path(self, config: Dict) -> Optional[str]:
        """Find the checkpoint file from multiple possible locations"""
        possible_paths = [
            config.get('checkpoint_path', ''),
            os.path.join(current_dir, 'pressure_abupt_checkpoint.th'),
            os.path.join(current_dir, 'surface_abupt_checkpoint.th'),
            os.environ.get("CARBENCH_ABUPT_STAR_CHECKPOINT", ""),
            os.environ.get("CARBENCH_ABUPT_STAR_CHECKPOINT", ""),
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                self.logger.info(f"Found checkpoint at: {path}")
                return path
        
        self.logger.warning("No checkpoint file found, will use random initialization")
        return None
    
    # ============= Required BenchmarkModel Interface Methods =============
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for benchmark reporting"""
        info = {
            'name': 'AB-UPT',
            'type': 'Transformer-based',
            'description': 'Attention-Based Unifying Pressure Transformer for DrivAerNet',
            'paper': 'AB-UPT: Learning Aerodynamic Pressure Fields',
            'architecture': 'Encoder-Decoder with Geometric Attention',
            'num_parameters': self.count_parameters() if self.model else 6006532,
            'input_features': ['surface_position', 'geometry'],
            'output_features': ['surface_pressure'],
            'checkpoint': self.checkpoint_path if self.checkpoint_path else 'None'
        }
        return info
    
    def load_model(self) -> nn.Module:
        """Load and return the model"""
        if self.model is None:
            self.setup()
        return self.model
    
    def get_data_loaders(self) -> Tuple[Any, Any, Any]:
        """Get all data loaders"""
        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            self.setup()
        
        # Return all three separate loaders for proper evaluation
        return self.train_loader, self.val_loader, self.test_loader
    
    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess batch and extract input/target pairs
        
        Args:
            batch: Batch dictionary from dataloader or standard (data, targets) tuple
            
        Returns:
            Tuple of (input_data, target_data in Pascal units)
        """
        # Handle standard (data, targets) format like other models
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, targets = batch
            
            # Handle shape requirements
            if data.dim() == 4 and data.shape[1] == 1:
                data = data.squeeze(1)
            if targets.dim() == 3 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
                
            return data, targets
        
        # Handle AB-UPT specific dict format
        if not isinstance(batch, dict):
            return batch, None
        
        # Extract target pressure (normalized by collator)
        target_pressure_normalized = batch.get("surface_query_pressure")
        if target_pressure_normalized is None:
            target_pressure_normalized = batch.get("surface_anchor_pressure")
        
        # Denormalize targets to Pascal units for metrics computation
        if target_pressure_normalized is not None:
            # The collator normalizes data, so we denormalize for evaluation metrics
            target_pressure_pascal = target_pressure_normalized * self.pressure_std + self.pressure_mean
            self.logger.debug(f"Target pressure (Pascal): mean={target_pressure_pascal.mean():.2f}, std={target_pressure_pascal.std():.2f}")
        else:
            target_pressure_pascal = None
        
        # Store full batch for forward pass
        self._current_batch = batch
        
        # Return actual input tensor shape for computational metrics
        if "surface_anchor_position" in batch:
            input_data = batch["surface_anchor_position"]
        elif "geometry_position" in batch:
            input_data = batch["geometry_position"]
        else:
            # Fallback to a properly shaped tensor
            input_data = torch.randn(self.batch_size, self.num_surface_anchor_points, 3, device=self.device)
        
        return input_data, target_pressure_pascal
    
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model
        
        Args:
            model: The model to use for forward pass
            x: Input tensor (for standard models) or None (for AB-UPT using cached batch)
            
        Returns:
            Model predictions in Pascal units
        """
        model.eval()
        
        # For AB-UPT, use the cached batch from preprocess_batch
        if hasattr(self, '_current_batch') and self._current_batch is not None:
            batch = self._current_batch
            
            with torch.no_grad():
                # Move batch to device
                batch_gpu = self._move_batch_to_device(batch)
                
                # Prepare input batch (remove targets)
                input_batch = self._prepare_input_batch(batch_gpu)
                
                # Forward pass with mixed precision
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                                   dtype=torch.float16):
                    predictions = model(**input_batch)
                
                # Extract pressure predictions (these are normalized)
                pred_pressure_normalized = self._extract_pressure_predictions(predictions)
                
                # Denormalize predictions to Pascal units for metrics
                pred_pressure_pascal = pred_pressure_normalized * self.pressure_std + self.pressure_mean
                
                self.logger.debug(f"Predicted pressure (Pascal): mean={pred_pressure_pascal.mean():.2f}, std={pred_pressure_pascal.std():.2f}")
                
                # Verify the output is in reasonable Pascal range
                if abs(pred_pressure_pascal.mean() + 93.41) > 200:
                    self.logger.warning(f"Predictions may be incorrect! Mean: {pred_pressure_pascal.mean():.2f} Pa")
                
                return pred_pressure_pascal
        else:
            # Fallback to standard forward pass for compatibility
            with torch.no_grad():
                output = model(x)
                return output
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        batch_gpu = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_gpu[key] = value.to(self.device)
            else:
                batch_gpu[key] = value
        return batch_gpu
    
    def _prepare_input_batch(self, batch: Dict) -> Dict:
        """Prepare input batch by removing target values"""
        input_batch = {}
        target_keys = ['surface_anchor_pressure', 'surface_anchor_wallshearstress',
                      'surface_query_pressure', 'surface_query_wallshearstress']
        
        for key, value in batch.items():
            if key not in target_keys:
                input_batch[key] = value
        
        return input_batch
    
    def _extract_pressure_predictions(self, predictions: Dict) -> torch.Tensor:
        """Extract pressure predictions from model output"""
        if isinstance(predictions, torch.Tensor):
            return predictions
        
        # Try different keys
        for key in ['surface_query_pressure', 'surface_anchor_pressure', 'pressure']:
            if key in predictions:
                return predictions[key]
        
        # Return first tensor found
        for value in predictions.values():
            if torch.is_tensor(value):
                return value
        
        # Fallback
        self.logger.error("Could not extract pressure predictions from model output!")
        return torch.zeros(1, device=self.device)
    
    def get_loss_function(self) -> nn.Module:
        """Get the loss function that expects inputs in Pascal units"""
        class DenormalizedMSELoss(nn.Module):
            """MSE loss that handles Pascal-scale inputs"""
            def __init__(self, mean, std):
                super().__init__()
                self.mean = mean
                self.std = std
                self.mse = nn.MSELoss()
            
            def forward(self, pred_pascal, target_pascal):
                # Both inputs are in Pascal units, normalize for loss computation
                pred_norm = (pred_pascal - self.mean) / self.std
                target_norm = (target_pascal - self.mean) / self.std
                return self.mse(pred_norm, target_norm)
        
        return DenormalizedMSELoss(self.pressure_mean, self.pressure_std)
    
    def get_normalization_functions(self) -> Tuple[Callable, Callable]:
        """Get normalization and denormalization functions
        
        Returns:
            Tuple of (normalize_fn, denormalize_fn)
        """
        def normalize(x):
            """Normalize from Pascal to normalized units"""
            return (x - self.pressure_mean) / self.pressure_std
        
        def denormalize(x):
            """Denormalize from normalized to Pascal units"""
            return x * self.pressure_std + self.pressure_mean
        
        return normalize, denormalize
    
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer for training"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """Get learning rate scheduler"""
        if self.config.get('epochs', 0) > 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        return None
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'epochs': self.config.get('epochs', 0),
            'batch_size': self.batch_size,
            'learning_rate': self.config.get('lr', 1e-4),
            'weight_decay': self.config.get('weight_decay', 1e-4),
            'early_stopping_patience': self.config.get('early_stopping_patience', 20),
            'gradient_clip': self.config.get('gradient_clip', 1.0),
            'mixed_precision': True,
            'num_workers': self.num_workers
        }
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """Load checkpoint into model"""
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    # Assume the checkpoint is the state dict itself
                    model.load_state_dict(checkpoint, strict=False)
            else:
                # Direct state dict
                model.load_state_dict(checkpoint, strict=False)
            
            self.logger.info("Checkpoint loaded successfully")
        else:
            self.logger.warning(f"Checkpoint not found at {checkpoint_path}")
        
        return model
    
    # ============= Setup and Initialization =============
    
    def setup(self):
        """Complete setup of model, datasets, and data loaders"""
        self.logger.info("Setting up AB-UPT model for evaluation...")
        
        # Initialize datasets first (needed for normalization stats)
        self._setup_datasets()
        
        # Compute or verify normalization statistics
        self._verify_normalization_statistics()
        
        # Initialize model
        self._setup_model()
        
        # Initialize collator with proper dataset
        self._setup_collator()
        
        # Create data loaders for all splits
        self._setup_dataloaders()
        
        self.logger.info(f"Setup complete. Model has {self.count_parameters():,} parameters")
    
    def _setup_model(self):
        """Initialize and load the model"""
        self.model = SurfaceOnlyABUPT().to(self.device)
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.model = self.load_checkpoint(self.checkpoint_path, self.model)
        else:
            self.logger.warning("No checkpoint loaded, using random initialization")
        
        self.model.eval()
    
    def _setup_datasets(self):
        """Initialize datasets for ALL splits"""
        self.logger.info(f"Loading datasets from {self.dataset_path}")
        
        # Load ALL three splits for proper evaluation
        try:
            self.logger.info("Loading train split...")
            self.dataset_train = DrivAerNetDataset(
                root=str(self.dataset_path),
                split="train",
                skip_missing=True
            )
            self.logger.info(f"Train dataset loaded: {len(self.dataset_train)} samples")
        except Exception as e:
            self.logger.warning(f"Could not load train split: {e}")
            self.dataset_train = None
        
        try:
            self.logger.info("Loading validation split...")
            self.dataset_val = DrivAerNetDataset(
                root=str(self.dataset_path),
                split="val",
                skip_missing=True
            )
            self.logger.info(f"Val dataset loaded: {len(self.dataset_val)} samples")
        except Exception as e:
            self.logger.warning(f"Could not load val split: {e}")
            self.dataset_val = None
        
        try:
            self.logger.info("Loading test split...")
            self.dataset_test = DrivAerNetDataset(
                root=str(self.dataset_path),
                split="test",
                skip_missing=True
            )
            self.logger.info(f"Test dataset loaded: {len(self.dataset_test)} samples")
        except Exception as e:
            self.logger.error(f"Could not load test split: {e}")
            raise
    
    def _verify_normalization_statistics(self):
        """Verify normalization statistics by sampling from datasets"""
        self.logger.info("Verifying normalization statistics...")
        
        sample_pressures = []
        datasets = [self.dataset_train, self.dataset_val, self.dataset_test]
        
        for dataset in datasets:
            if dataset is not None and len(dataset) > 0:
                # Sample a few examples to check statistics
                for i in range(min(5, len(dataset))):
                    try:
                        pressure = dataset.getitem_surface_pressure(i)
                        sample_pressures.append(pressure.flatten())
                    except Exception as e:
                        self.logger.warning(f"Could not load sample {i}: {e}")
        
        if sample_pressures:
            all_pressures = torch.cat(sample_pressures)
            actual_mean = all_pressures.mean().item()
            actual_std = all_pressures.std().item()
            
            self.logger.info(f"Dataset statistics - Mean: {actual_mean:.2f} Pa, Std: {actual_std:.2f} Pa")
            self.logger.info(f"Expected statistics - Mean: {self.pressure_mean:.2f} Pa, Std: {self.pressure_std:.2f} Pa")
            
            # Check if statistics match expected values
            if abs(actual_mean - self.pressure_mean) > 20 or abs(actual_std - self.pressure_std) > 20:
                self.logger.warning("Dataset statistics differ significantly from expected values!")
                # Optionally update statistics
                # self.pressure_mean = actual_mean
                # self.pressure_std = actual_std
    
    def _setup_collator(self):
        """Initialize the collator with the appropriate dataset"""
        # Use the first available dataset for collator initialization
        dataset_for_collator = self.dataset_train or self.dataset_val or self.dataset_test
        
        if dataset_for_collator is None:
            raise ValueError("No dataset available for collator initialization")
        
        self.collator = SurfaceOnlyAbuptCollator(
            num_geometry_points=self.num_geometry_points,
            num_surface_anchor_points=self.num_surface_anchor_points,
            num_geometry_supernodes=self.num_geometry_supernodes,
            use_query_positions=True,
            dataset=dataset_for_collator
        )
        
        self.logger.info(f"Collator initialized with {dataset_for_collator.__class__.__name__}")
    
    def _setup_dataloaders(self):
        """Create data loaders for ALL splits"""
        self.logger.info("Creating data loaders for all splits...")
        
        # Create train loader
        if self.dataset_train is not None:
            self.train_loader = torch.utils.data.DataLoader(
                range(len(self.dataset_train)),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda indices: self._collate_fn(indices, self.dataset_train),
                num_workers=0,
                pin_memory=True
            )
            self.logger.info(f"Train loader created: {len(self.train_loader)} batches")
        else:
            self.train_loader = None
        
        # Create validation loader
        if self.dataset_val is not None:
            self.val_loader = torch.utils.data.DataLoader(
                range(len(self.dataset_val)),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda indices: self._collate_fn(indices, self.dataset_val),
                num_workers=0,
                pin_memory=True
            )
            self.logger.info(f"Val loader created: {len(self.val_loader)} batches")
        else:
            self.val_loader = None
        
        # Create test loader
        if self.dataset_test is not None:
            self.test_loader = torch.utils.data.DataLoader(
                range(len(self.dataset_test)),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda indices: self._collate_fn(indices, self.dataset_test),
                num_workers=0,
                pin_memory=True
            )
            self.logger.info(f"Test loader created: {len(self.test_loader)} batches")
        else:
            raise ValueError("Test dataset is required for evaluation")
    
    def _collate_fn(self, indices: List[int], dataset: DrivAerNetDataset) -> Dict:
        """Custom collate function for AB-UPT
        
        Args:
            indices: Batch indices
            dataset: The specific dataset to load from (train/val/test)
        """
        batch_samples = []
        
        for idx in indices:
            try:
                sample = {
                    'surface_position_vtp': dataset.getitem_surface_position_vtp(idx),
                    'surface_pressure': dataset.getitem_surface_pressure(idx),
                    'surface_wallshearstress': dataset.getitem_surface_wallshearstress(idx),
                }
                batch_samples.append(sample)
            except Exception as e:
                self.logger.error(f"Error loading sample {idx}: {e}")
                # Create dummy sample to maintain batch size
                sample = {
                    'surface_position_vtp': torch.randn(10000, 3),
                    'surface_pressure': torch.randn(10000, 1),
                    'surface_wallshearstress': torch.randn(10000, 3),
                }
                batch_samples.append(sample)
        
        return self.collator(batch_samples)
    
    # ============= Utility Methods =============
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Calculate model size in MB"""
        if self.model is None:
            return 0.0
        
        param_size = sum(
            param.nelement() * param.element_size()
            for param in self.model.parameters()
        )
        buffer_size = sum(
            buffer.nelement() * buffer.element_size()
            for buffer in self.model.buffers()
        )
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_training_history(self) -> Dict:
        """Get training history"""
        return self.training_history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        if self.model is not None:
            torch.save({
                'state_dict': self.model.state_dict(),
                'config': self.config,
                'training_history': self.training_history,
                'normalization': {
                    'pressure_mean': self.pressure_mean,
                    'pressure_std': self.pressure_std
                }
            }, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")