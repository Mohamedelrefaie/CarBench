# benchmark/base_model.py
"""
Base model interface that all benchmark models must implement.
This allows models to keep their own structure while providing a unified interface.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Callable
from torch.utils.data import DataLoader


class BenchmarkModel(ABC):
    """
    Abstract base class for benchmark models.
    Each model must implement this interface through a wrapper.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model wrapper.
        
        Args:
            config: Configuration dictionary containing model and data settings
        """
        self.config = config
        self.model = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Load and return the actual model.
        This should import and instantiate the model from the model's own code.
        
        Returns:
            The PyTorch model
        """
        pass
    
    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test data loaders using the model's own data loading code.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> nn.Module:
        """
        Get the loss function for this model.
        
        Returns:
            Loss function
        """
        pass
    
    @abstractmethod
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Get the optimizer for this model.
        
        Args:
            model: The model to optimize
            
        Returns:
            Optimizer
        """
        pass
    
    @abstractmethod
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[object]:
        """
        Get the learning rate scheduler.
        
        Args:
            optimizer: The optimizer
            
        Returns:
            Scheduler or None
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a batch of data.
        This handles any model-specific data formatting.
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        pass
    
    @abstractmethod
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Handles any model-specific forward pass requirements.
        
        Args:
            model: The model
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_normalization_functions(self) -> Tuple[Optional[Callable], Optional[Callable]]:
        """
        Get normalization and denormalization functions.
        
        Returns:
            Tuple of (normalize_fn, denormalize_fn)
        """
        return None, None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.__class__.__name__.replace('Wrapper', ''),
            'config': self.config
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Returns:
            Dictionary with training settings
        """
        return {
            'epochs': self.config.get('epochs', 150),
            'batch_size': self.config.get('batch_size', 12),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'early_stopping_patience': self.config.get('early_stopping_patience', 20),
            'gradient_clip': self.config.get('gradient_clip', 1.0)
        }
