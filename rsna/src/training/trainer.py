import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional
from collections import OrderedDict
import torch.nn.functional as F
from pathlib import Path

from ..core import Config
from ..models import WeightedBCELoss

class ModelTrainer:
    """Training class for intracranial hemorrhage detection models"""
    
    def __init__(self, model: nn.Module, device: str, config: Config):
        self.model = model
        self.device = device
        self.config = config
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def calculate_metric(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate weighted BCE metric"""
        preds_tensor = torch.from_numpy(predictions)
        targets_tensor = torch.from_numpy(targets)
        weight = torch.ones_like(preds_tensor)
        weight[:, 0] += 1  # Weight the 'any' class more heavily
        
        metric = F.binary_cross_entropy_with_logits(
            preds_tensor, targets_tensor, weight=weight, reduction='mean'
        )
        return metric.item()
    
    def validate_epoch(self, valid_loader: torch.utils.data.DataLoader, 
                      loss_fn: nn.Module) -> Tuple[float, float]:
        """Validate model for one epoch"""
        self.model.eval()
        losses = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].float().to(self.device)
                labels = batch['label'].float().squeeze().to(self.device)
                
                predictions = self.model(images)
                loss = loss_fn(predictions, labels).mean()
                
                losses.append(loss.item())
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        mean_loss = np.mean(losses)
        metric = self.calculate_metric(all_predictions, all_labels)
        
        self.model.train()
        return mean_loss, metric
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   use_amp: bool = False) -> Tuple[float, float]:
        """Train model for one epoch"""
        self.model.train()
        losses = []
        all_labels = []
        all_predictions = []
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].float().to(self.device)
            labels = batch['label'].float().squeeze().to(self.device)
            
            optimizer.zero_grad()
            
            if use_amp:
                try:
                    from apex import amp
                    predictions = self.model(images)
                    loss = loss_fn(predictions, labels).mean()
                    scaled_loss = amp.scale_loss(loss, optimizer)
                    scaled_loss.backward()
                except ImportError:
                    predictions = self.model(images)
                    loss = loss_fn(predictions, labels).mean()
                    loss.backward()
            else:
                predictions = self.model(images)
                loss = loss_fn(predictions, labels).mean()
                loss.backward()
            
            optimizer.step()
            
            losses.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            
            if batch_idx % 100 == 0:
                current_loss = np.mean(losses[-100:])
                progress_bar.set_postfix({'loss': f'{current_loss:.5f}'})
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        mean_loss = np.mean(losses)
        metric = self.calculate_metric(all_predictions, all_labels)
        
        if scheduler is not None:
            scheduler.step()
        
        return mean_loss, metric
    
    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        save_path = self.config.MODEL_DIR / f"model_{epoch}.pt"
        torch.save(checkpoint, save_path)
        
        if loss < self.best_loss:
            self.best_loss = loss
            best_save_path = self.config.MODEL_DIR / "best_model.pt"
            torch.save(checkpoint, best_save_path)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: nn.Module,
              num_epochs: int,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              use_amp: bool = False,
              patience: int = 5) -> Dict[str, Any]:
        """Complete training loop"""
        
        training_history = {
            'train_loss': [],
            'train_metric': [],
            'valid_loss': [],
            'valid_metric': [],
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_loss, train_metric = self.train_epoch(
                train_loader, optimizer, loss_fn, scheduler, use_amp
            )
            
            valid_loss, valid_metric = self.validate_epoch(valid_loader, loss_fn)
            
            training_history['train_loss'].append(train_loss)
            training_history['train_metric'].append(train_metric)
            training_history['valid_loss'].append(valid_loss)
            training_history['valid_metric'].append(valid_metric)
            
            print(f"Train Loss: {train_loss:.5f}, Train Metric: {train_metric:.5f}")
            print(f"Valid Loss: {valid_loss:.5f}, Valid Metric: {valid_metric:.5f}")
            
            self.save_checkpoint(epoch, valid_loss, optimizer, scheduler)
            
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        return training_history

def train_fold(fold_idx: int, config: Config, model_name: str = 'resnext101') -> None:
    """Train model for a specific fold"""
    from ..data import create_data_loaders
    from ..models import create_model
    from ..training import create_optimizer, create_scheduler
    from ..models import WeightedBCELoss
    
    print(f"Training fold {fold_idx}")
    
    train_loader, valid_loader = create_data_loaders(fold_idx, config)
    
    model = create_model(model_name, config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    optimizer = create_optimizer(
        model, 
        optimizer_name='radam', 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = create_scheduler(optimizer, scheduler_name='step')
    
    loss_fn = WeightedBCELoss()
    
    trainer = ModelTrainer(model, config.DEVICE, config)
    
    try:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2", 
            keep_batchnorm_fp32=True, verbosity=0
        )
        use_amp = True
    except ImportError:
        print("Warning: Apex not available, training without mixed precision")
        use_amp = False
    
    training_history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=config.NUM_EPOCHS,
        scheduler=scheduler,
        use_amp=use_amp,
        patience=5
    )
    
    model = model.cpu()
    del model, optimizer, scheduler, trainer
    torch.cuda.empty_cache()
    
    print(f"Completed training fold {fold_idx}")
    
    return training_history
