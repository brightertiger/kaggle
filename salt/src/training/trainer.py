import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from ..core.config import Config
from ..models.models import create_model
from ..models.loss import create_loss_function, create_metric_function

class ModelTrainer:
    """Model training class for Salt Identification"""
    
    def __init__(self, fold_idx: int, config: Config):
        self.fold_idx = fold_idx
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Training history
        self.train_losses = []
        self.valid_losses = []
        self.train_metrics = []
        self.valid_metrics = []
        
        # Model paths
        self.model_path = config.MODEL_DIR / f"model_{fold_idx}.pth"
        
    def create_model_and_optimizer(self, model_name: str) -> Tuple[nn.Module, optim.Optimizer]:
        """Create model and optimizer for training"""
        model = create_model(model_name, self.config)
        model = model.to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        return model, optimizer
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, metric: float):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metric': metric,
            'config': self.config.to_dict()
        }
        torch.save(state, self.model_path)
        print(f"Model saved at epoch {epoch} with metric {metric:.4f}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None) -> int:
        """Load model checkpoint"""
        if not self.model_path.exists():
            print("No checkpoint found, starting from scratch")
            return 0
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        epoch = checkpoint.get('epoch', 0)
        metric = checkpoint.get('metric', 0.0)
        print(f"Model loaded from epoch {epoch} with metric {metric:.4f}")
        
        return epoch
    
    def adjust_learning_rate(self, optimizer: optim.Optimizer, factor: float = 0.5):
        """Adjust learning rate"""
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor
        print(f"Learning rate reduced by factor {factor}")
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, 
                   metric_fn: callable) -> Tuple[float, float]:
        """Train model for one epoch"""
        model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        for batch_idx, sample in enumerate(train_loader):
            images = sample['image'].to(self.device)
            masks = sample['mask'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metric = metric_fn(predictions, masks)
                total_loss += loss.item()
                total_metric += metric
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def validate_epoch(self, model: nn.Module, valid_loader: DataLoader, 
                      criterion: nn.Module, metric_fn: callable) -> Tuple[float, float]:
        """Validate model for one epoch"""
        model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(valid_loader):
                images = sample['image'].to(self.device)
                masks = sample['mask'].to(self.device)
                
                # Forward pass
                predictions = model(images)
                loss = criterion(predictions, masks)
                
                # Calculate metrics
                metric = metric_fn(predictions, masks)
                total_loss += loss.item()
                total_metric += metric
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def train(self, model_name: str, train_loader: DataLoader, valid_loader: DataLoader,
              resume: bool = False) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training for fold {self.fold_idx}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        
        # Create model and optimizer
        model, optimizer = self.create_model_and_optimizer(model_name)
        
        # Create loss and metric functions
        criterion = create_loss_function(self.config.LOSS_TYPE, {
            'dice_weight': self.config.DICE_WEIGHT,
            'bce_weight': self.config.BCE_WEIGHT
        })
        metric_fn = create_metric_function(self.config.METRIC_TYPE, {
            'cutoff': self.config.IOU_CUTOFF,
            'squash': self.config.IOU_SQUASH
        })
        
        # Load checkpoint if resuming
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint(model, optimizer)
        
        # Training loop
        best_metric = 0.78  # Baseline metric
        lr_patience_counter = 0
        early_stop_counter = 0
        
        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_loss, train_metric = self.train_epoch(
                model, train_loader, optimizer, criterion, metric_fn
            )
            
            # Validate
            valid_loss, valid_metric = self.validate_epoch(
                model, valid_loader, criterion, metric_fn
            )
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_metrics.append(train_metric)
            self.valid_metrics.append(valid_metric)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
            print(f"Valid Loss: {valid_loss:.4f}, Valid Metric: {valid_metric:.4f}")
            
            # Save best model
            if valid_metric > best_metric:
                self.save_checkpoint(model, optimizer, epoch, valid_metric)
                best_metric = valid_metric
                early_stop_counter = 0
                lr_patience_counter = 0
            else:
                early_stop_counter += 1
                lr_patience_counter += 1
                print(f"Model hasn't improved in {early_stop_counter} epochs")
            
            # Learning rate reduction
            if lr_patience_counter >= self.config.LR_REDUCTION_PATIENCE:
                self.adjust_learning_rate(optimizer, self.config.LR_REDUCTION_FACTOR)
                lr_patience_counter = 0
            
            # Early stopping
            if early_stop_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
        
        print(f"\nTraining completed for fold {self.fold_idx}")
        print(f"Best validation metric: {best_metric:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_metrics': self.train_metrics,
            'valid_metrics': self.valid_metrics,
            'best_metric': best_metric
        }
