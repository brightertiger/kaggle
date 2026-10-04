import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

from .models import CenterLoss
from torch.utils.data import DataLoader

class ModelCheckpoint:
    def __init__(self, save_dir: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = float("inf") if mode == 'min' else -float("inf")
        
        os.makedirs(save_dir, exist_ok=True)
    
    def __call__(self, epoch: int, model: nn.Module, score: float, 
                 optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None):
        if self.save_best_only:
            if self._is_better_score(score):
                self.best_score = score
                self._save_checkpoint(epoch, model, score, optimizer, scheduler)
        else:
            self._save_checkpoint(epoch, model, score, optimizer, scheduler)
    
    def _is_better_score(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def _save_checkpoint(self, epoch: int, model: nn.Module, score: float,
                        optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': float(score),
            'model_kwargs': model.model_kwargs,
            'class_names': model.class_names,
            'model_type': 'siamese' if hasattr(model, 'score_embeddings') else 'classification',
            'image_size': getattr(model, 'image_size', None),
        }
        
        if getattr(self, 'center_loss', None) is not None:
            checkpoint['center_loss_state_dict'] = self.center_loss.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.save_dir, 'model.pth'))
        print(f'Model saved with {self.monitor}: {score:.4f}')

class Trainer:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module, metric_fn: nn.Module, 
                   scheduler: Optional[Any] = None, center_loss: Optional[CenterLoss] = None,
                   center_loss_weight: float = 0.0) -> Tuple[float, float]:
        self.model.train()
        losses = []
        metrics = []
        sizes = []
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].reshape(-1).to(self.device)
            
            optimizer.zero_grad()
            
            if center_loss is not None:
                preds, embeddings = self.model(images)
                classification_loss = loss_fn(preds, labels)
                center_loss_val = center_loss(embeddings, labels)
                total_loss = classification_loss + center_loss_weight * center_loss_val
            else:
                output = self.model(images)
                preds = output[0] if isinstance(output, tuple) else output
                total_loss = loss_fn(preds, labels)
            
            total_loss.backward()
            optimizer.step()
            
            
            metric = metric_fn(preds, labels)
            losses.append(total_loss.item())
            metrics.append(metric.item())
            sizes.append(labels.shape[0])
            
            pbar.set_postfix({
                'loss': f'{np.average(losses, weights=sizes):.4f}',
                'metric': f'{np.average(metrics, weights=sizes):.4f}'
            })
        
        return float(np.average(losses, weights=sizes)), float(np.average(metrics, weights=sizes))
    
    def validate(self, val_loader: DataLoader, loss_fn: nn.Module, 
                metric_fn: nn.Module, center_loss: Optional[CenterLoss] = None,
                center_loss_weight: float = 0.0) -> Tuple[float, float]:
        self.model.eval()
        losses = []
        metrics = []
        sizes = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].reshape(-1).to(self.device)
                
                if center_loss is not None:
                    preds, embeddings = self.model(images)
                    classification_loss = loss_fn(preds, labels)
                    center_loss_val = center_loss(embeddings, labels)
                    total_loss = classification_loss + center_loss_weight * center_loss_val
                else:
                    output = self.model(images)
                    preds = output[0] if isinstance(output, tuple) else output
                    total_loss = loss_fn(preds, labels)
                
                metric = metric_fn(preds, labels)
                losses.append(total_loss.item())
                metrics.append(metric.item())
                sizes.append(labels.shape[0])
                
                pbar.set_postfix({
                    'loss': f'{np.average(losses, weights=sizes):.4f}',
                    'metric': f'{np.average(metrics, weights=sizes):.4f}'
                })
        
        return float(np.average(losses, weights=sizes)), float(np.average(metrics, weights=sizes))
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
              optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
              metric_fn: nn.Module, num_epochs: int, 
              checkpoint: Optional[ModelCheckpoint] = None,
              scheduler: Optional[Any] = None,
              center_loss: Optional[CenterLoss] = None,
              center_loss_weight: float = 0.0) -> Dict[str, list]:
        
        self.train_losses, self.val_losses = [], []
        self.train_metrics, self.val_metrics = [], []
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training
            train_loss, train_metric = self.train_epoch(
                train_loader, optimizer, loss_fn, metric_fn, 
                scheduler, center_loss, center_loss_weight
            )
            
            if scheduler is not None:
                scheduler.step()

            # Validation
            if val_loader is not None:
                val_loss, val_metric = self.validate(
                    val_loader, loss_fn, metric_fn, 
                    center_loss, center_loss_weight
                )
            else:
                val_loss, val_metric = 0.0, 0.0
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metric)
            self.val_metrics.append(val_metric)
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}')
            if val_loader is not None:
                print(f'Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}')
            
            # Save checkpoint
            if checkpoint is not None:
                score = val_loss if val_loader is not None else train_loss
                checkpoint(epoch, self.model, score, optimizer, scheduler)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

class SiameseTrainer:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module, metric_fn: nn.Module,
                   scheduler: Optional[Any] = None) -> Tuple[float, float]:
        self.model.train()
        losses = []
        metrics = []
        sizes = []
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch in pbar:
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            preds = self.model(image1, image2)
            loss = loss_fn(preds, labels)
            
            loss.backward()
            optimizer.step()
            
            
            metric = metric_fn(preds, labels)
            losses.append(loss.item())
            metrics.append(metric.item())
            sizes.append(labels.shape[0])
            
            pbar.set_postfix({
                'loss': f'{np.average(losses, weights=sizes):.4f}',
                'metric': f'{np.average(metrics, weights=sizes):.4f}'
            })
        
        return float(np.average(losses, weights=sizes)), float(np.average(metrics, weights=sizes))
    
    def validate(self, val_loader: DataLoader, loss_fn: nn.Module, 
                metric_fn: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        losses = []
        metrics = []
        sizes = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for batch in pbar:
                image1 = batch['image1'].to(self.device)
                image2 = batch['image2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                preds = self.model(image1, image2)
                loss = loss_fn(preds, labels)
                metric = metric_fn(preds, labels)
                
                losses.append(loss.item())
                metrics.append(metric.item())
                sizes.append(labels.shape[0])
                
                pbar.set_postfix({
                    'loss': f'{np.average(losses, weights=sizes):.4f}',
                    'metric': f'{np.average(metrics, weights=sizes):.4f}'
                })
        
        return float(np.average(losses, weights=sizes)), float(np.average(metrics, weights=sizes))
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
              optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
              metric_fn: nn.Module, num_epochs: int, 
              checkpoint: Optional[ModelCheckpoint] = None,
              scheduler: Optional[Any] = None) -> Dict[str, list]:
        
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training
            train_loss, train_metric = self.train_epoch(
                train_loader, optimizer, loss_fn, metric_fn, scheduler
            )
            
            if scheduler is not None:
                scheduler.step()

            # Validation
            if val_loader is not None:
                val_loss, val_metric = self.validate(val_loader, loss_fn, metric_fn)
            else:
                val_loss, val_metric = 0.0, 0.0
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}')
            if val_loader is not None:
                print(f'Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}')
            
            # Save checkpoint
            if checkpoint is not None:
                score = val_loss if val_loader is not None else train_loss
                checkpoint(epoch, self.model, score, optimizer, scheduler)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
