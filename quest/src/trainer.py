import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional

try:
    from apex import amp
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Warning: Apex not available. Mixed precision training disabled.")

from .models import ModelFactory
from .data_utils import DataProcessor


class Trainer:
    """Training class for question understanding models"""
    
    def __init__(self, config, model_type: str = "question_understanding"):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ModelFactory.create_model(model_type, config)
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            min_lr=1e-7,
            patience=1,
            verbose=True
        )
        
        # Mixed precision training
        if AMP_AVAILABLE:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer,
                opt_level="O2",
                keep_batchnorm_fp32=True,
                verbosity=0
            )
        
        self.data_processor = DataProcessor(config)
        
    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce loss across batch and labels"""
        batch_size = loss.shape[0]
        return loss.sum() / (batch_size * self.config.num_labels)
    
    def evaluate_model(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """Evaluate model using Spearman correlation"""
        correlations = []
        
        for idx in range(self.config.num_labels):
            label = labels[:, idx]
            score = scores[:, idx]
            
            # Add small noise to avoid ties
            score = score + np.random.normal(0, 1e-7, score.shape[0])
            
            correlation = spearmanr(label, score).correlation
            if not np.isnan(correlation):
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def validate_model(self, valid_loader) -> float:
        """Validate model and return validation loss"""
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                # Move to device
                question = batch['question'].to(self.device)
                answer = batch['answer'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(question, answer)
                scores = torch.sigmoid(logits)
                
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Combine all predictions
        all_scores = np.vstack(all_scores)
        all_labels = np.vstack(all_labels)
        
        # Calculate validation metric (negative correlation for minimization)
        val_metric = -self.evaluate_model(all_labels, all_scores)
        
        self.model.train()
        return val_metric
    
    def save_model(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            question = batch['question'].to(self.device)
            answer = batch['answer'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(question, answer)
            loss = self.criterion(logits, labels)
            loss = self.reduce_loss(loss)
            
            # Backward pass with mixed precision
            if AMP_AVAILABLE:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss / num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def train_fold(self, fold: int, data_dir: str, model_dir: str) -> float:
        """Train model for a specific fold"""
        print(f"Training fold {fold}...")
        
        # Create data loaders
        train_loader, valid_loader = self.data_processor.create_data_loaders(
            fold, data_dir, self.config.batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3
        
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_model(valid_loader)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model_path = os.path.join(model_dir, f"fold_{fold}", "best_model.pt")
                self.save_model(model_path, epoch, val_loss)
                
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return best_val_loss
    
    def train_all_folds(self, data_dir: str, model_dir: str):
        """Train model for all folds"""
        fold_results = []
        
        for fold in range(1, self.config.n_folds + 1):
            val_loss = self.train_fold(fold, data_dir, model_dir)
            fold_results.append(val_loss)
            print(f"Fold {fold} completed with val_loss: {val_loss:.4f}")
        
        print(f"All folds completed. Average val_loss: {np.mean(fold_results):.4f}")
        return fold_results
