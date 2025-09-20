import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple
import os

class PronounTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 loss_fn: nn.Module = None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
    
    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                tokens, offsets, feature_a, feature_b, labels = [x.to(self.device) for x in batch]
                
                outputs = self.model(tokens, offsets, feature_a, feature_b)
                loss = self.loss_fn(outputs, labels.squeeze())
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            tokens, offsets, feature_a, feature_b, labels = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()
            outputs = self.model(tokens, offsets, feature_a, feature_b)
            loss = self.loss_fn(outputs, labels.squeeze())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int,
              save_dir: str) -> Dict[str, Any]:
        
        os.makedirs(save_dir, exist_ok=True)
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_dir, epoch, val_loss)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
        
        return history
    
    def save_checkpoint(self, save_dir: str, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
        print(f'Model saved with validation loss: {val_loss:.4f}')
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']
