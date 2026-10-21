# benchmark/trainer.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Unified trainer that works with model wrappers. Handles normalization consistently and tracks training history.
"""

import torch
import torch.nn as nn
import time
import os
from tqdm import tqdm
from typing import Optional, Any, Dict, List


class UnifiedTrainer:
    """Unified trainer for all benchmark models."""
    
    def __init__(
        self,
        model: nn.Module,
        wrapper: 'BenchmarkModel',
        evaluator: 'UnifiedEvaluator',
        save_dir: str
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            wrapper: Model wrapper
            evaluator: Evaluator instance for metrics tracking
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.wrapper = wrapper
        self.evaluator = evaluator
        self.save_dir = save_dir
        self.device = wrapper.device
        
        # Mixed precision setup
        self.use_amp = wrapper.config.get('mixed_precision', True) and self.device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        
        # Create checkpoint directory
        model_info = self.wrapper.get_model_info()
        original_model_name = model_info.get('name', '').lower()
        
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        epochs: int = 150,
        early_stopping_patience: int = 20,
        gradient_clip: Optional[float] = 1.0
    ):
        """
        Train the model with proper normalization handling.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            gradient_clip: Gradient clipping value
        """
        patience_counter = 0
        
        # Get loss function and normalization
        criterion = self.wrapper.get_loss_function()
        normalize_fn, denormalize_fn = self.wrapper.get_normalization_functions()
        
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Gradient clipping: {gradient_clip}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Normalization: {'Yes' if normalize_fn is not None else 'No'}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(
                train_loader, criterion, optimizer, 
                normalize_fn, gradient_clip
            )
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self._validate_epoch(
                val_loader, criterion, normalize_fn
            )
            self.val_losses.append(val_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Update scheduler
            if scheduler is not None:
                if hasattr(scheduler, 'step'):
                    if 'ReduceLROnPlateau' in scheduler.__class__.__name__:
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
            
            # Track metrics in evaluator
            epoch_time = time.time() - epoch_start_time
            self.evaluator.train_metrics.end_epoch(epoch, val_loss, current_lr)
            self.evaluator.update_training_history(epoch, train_loss, val_loss, current_lr)
            
            # Print progress
            improvement = ""
            if val_loss < self.best_val_loss:
                improvement = " * BEST"
            
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s{improvement}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint('best', epoch, val_loss, train_loss)
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\n Early stopping triggered at epoch {epoch+1}")
                print(f"   Best epoch: {self.best_epoch} with val loss: {self.best_val_loss:.6f}")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint('periodic', epoch, val_loss, train_loss)

        # Always keep the final-epoch weights (best_checkpoint is val-driven and
        # periodic saves are every 10 epochs, so the last epoch is otherwise lost)
        self._save_checkpoint('final', epoch, val_loss, train_loss)

        # Final summary
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final train loss: {self.train_losses[-1]:.6f}")
        print(f"Final val loss: {self.val_losses[-1]:.6f}")
        print(f"Total epochs: {len(self.train_losses)}")
        print(f"{'='*60}\n")
        
        # Load best model for final evaluation
        best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
        if os.path.exists(best_checkpoint_path):
            print("Loading best model for final evaluation...")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded from epoch {checkpoint['epoch'] + 1}")
    
    def _train_epoch(
        self,
        train_loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        normalize_fn: Optional[Any] = None,
        gradient_clip: Optional[float] = None
    ) -> float:
        """
        Train for one epoch with proper normalization.
        
        NORMALIZATION FLOW:
        1. Get raw data from dataloader (original scale)
        2. Normalize targets if model uses normalization
        3. Forward pass (model outputs in normalized scale)
        4. Compute loss in normalized scale
        5. Backpropagate
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {self.current_epoch + 1:3d}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Preprocess batch - gets raw data in original scale
            data, targets_original = self.wrapper.preprocess_batch(batch)
            data = data.to(self.device)
            targets_original = targets_original.to(self.device)
            
            # Normalize targets if model uses normalization
            if normalize_fn is not None:
                targets_for_loss = normalize_fn(targets_original)
            else:
                targets_for_loss = targets_original
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.wrapper.forward(self.model, data)
                    loss = criterion(outputs, targets_for_loss)
                
                self.scaler.scale(loss).backward()
                
                if gradient_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.wrapper.forward(self.model, data)
                loss = criterion(outputs, targets_for_loss)
                
                loss.backward()
                
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            total_loss += batch_loss * data.size(0)
            total_samples += data.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss, 'avg_loss': total_loss / total_samples})
        
        return total_loss / total_samples
    
    def _validate_epoch(
        self,
        val_loader: Any,
        criterion: nn.Module,
        normalize_fn: Optional[Any] = None
    ) -> float:
        """
        Validate for one epoch with proper normalization.
        
        Same normalization flow as training, but without backpropagation.
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {self.current_epoch + 1:3d}"):
                # Preprocess batch - gets raw data in original scale
                data, targets_original = self.wrapper.preprocess_batch(batch)
                data = data.to(self.device)
                targets_original = targets_original.to(self.device)
                
                # Normalize targets if model uses normalization
                if normalize_fn is not None:
                    targets_for_loss = normalize_fn(targets_original)
                else:
                    targets_for_loss = targets_original
                
                # Forward pass
                outputs = self.wrapper.forward(self.model, data)
                
                # Calculate loss in normalized scale
                loss = criterion(outputs, targets_for_loss)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        
        return total_loss / total_samples
    
    def _save_checkpoint(self, checkpoint_type: str, epoch: int, 
                        val_loss: float, train_loss: float):
        """Save model checkpoint with training history."""
        # Get model info
        model_info = self.wrapper.get_model_info().copy()
        
        # Ensure the model name is consistent
        if 'name' in model_info:
            original_name = str(self.wrapper.__class__.__name__).replace('Wrapper', '').lower()
            model_info['name'] = original_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'model_info': model_info,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates
            }
        }
        
        if checkpoint_type == 'best':
            filename = 'best_checkpoint.pth'
        elif checkpoint_type == 'periodic':
            filename = f'checkpoint_epoch_{epoch+1}.pth'
        else:
            filename = f'{checkpoint_type}_checkpoint.pth'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if checkpoint_type == 'best':
            print(f"   Best model saved to {filename}")