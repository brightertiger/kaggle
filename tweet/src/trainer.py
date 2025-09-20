import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.optimization import AdamW
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Tuple
from .config import Config
from .models import TweetSentimentModel, TweetLoss

class TweetTrainer:
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        
    def _setup_optimizer(self, model: nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
        exclude_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        params = list(model.named_parameters())
        param_groups = [
            {
                'params': [p for n, p in params if not any(ex in n for ex in exclude_params)],
                'weight_decay': self.config.model.weight_decay
            },
            {
                'params': [p for n, p in params if any(ex in n for ex in exclude_params)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(param_groups, lr=self.config.model.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.config.model.scheduler_factor,
            min_lr=self.config.model.scheduler_min_lr,
            patience=self.config.model.scheduler_patience
        )
        
        return optimizer, scheduler
    
    def _save_checkpoint(self, model: nn.Module, fold: int, epoch: int, loss: float):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.data.model_path, f'model_fold_{fold}.pt')
        torch.save(checkpoint, save_path)
    
    def _validate(self, model: nn.Module, valid_loader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                start_logits, end_logits, aux_logits = model(batch['tokens'], batch['masks'])
                loss = loss_fn(
                    start_logits, end_logits,
                    batch['start_idx'], batch['end_idx'],
                    aux_logits, batch['aux_label']
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                for key, value in batch.items():
                    batch[key] = value.to('cpu')
        
        model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train_fold(self, fold: int, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader):
        torch.cuda.empty_cache()
        
        model = TweetSentimentModel(self.config).to(self.device)
        optimizer, scheduler = self._setup_optimizer(model)
        loss_fn = TweetLoss(self.config)
        
        log_file = os.path.join(self.config.data.model_path, f'training_log_fold_{fold}.txt')
        if os.path.exists(log_file):
            os.remove(log_file)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.model.max_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f'Fold {fold}, Epoch {epoch+1}/{self.config.model.max_epochs}',
                leave=False
            )
            
            optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                start_logits, end_logits, aux_logits = model(batch['tokens'], batch['masks'])
                loss = loss_fn(
                    start_logits, end_logits,
                    batch['start_idx'], batch['end_idx'],
                    aux_logits, batch['aux_label']
                )
                
                loss = loss / self.config.model.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.model.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), self.config.model.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.model.gradient_accumulation_steps
                num_batches += 1
                
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'train_loss': f'{avg_loss:.4f}'})
                
                for key, value in batch.items():
                    batch[key] = value.to('cpu')
            
            val_loss = self._validate(model, valid_loader, loss_fn)
            scheduler.step(val_loss)
            
            progress_bar.set_postfix({
                'train_loss': f'{avg_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            
            log_message = f'Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}\n'
            with open(log_file, 'a') as f:
                f.write(log_message)
            
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint(model, fold, epoch, val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.training.early_stopping_patience:
                    print(f'Early stopping triggered for fold {fold}')
                    break
        
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        
        return best_loss
