import os
import random
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import fbeta_score
from typing import Dict, Optional

from .config import Config
from .models import ResNextClassifier, ModelFactory, save_model_checkpoint, load_model_checkpoint
from .data_utils import create_data_loaders


class MetricsCalculator:
    def __init__(self, config: Config):
        self.config = config
        self.thresholds = config.thresholds
        self.top_k = config.top_k
    
    def make_mask(self, argsorted: np.ndarray, top_n: int) -> np.ndarray:
        top_n = min(top_n, argsorted.shape[1])
        mask = np.zeros_like(argsorted, dtype=np.uint8)
        np.put_along_axis(mask, argsorted[:, -top_n:], 1, axis=1)
        return mask

    def binarize_predictions(self, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        argsorted = probabilities.argsort(axis=1)
        max_mask = self.make_mask(argsorted, self.top_k)
        min_mask = self.make_mask(argsorted, 1)
        prob_mask = probabilities > threshold
        return (max_mask & prob_mask) | min_mask
    
    def calculate_fbeta_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            threshold: float = 0.2) -> float:
        y_pred_binary = self.binarize_predictions(y_pred, threshold)
        return fbeta_score(np.asarray(y_true) > 0.5, y_pred_binary,
                           beta=self.config.f2_beta, average='samples', zero_division=0)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}
        for threshold in self.thresholds:
            fbeta = self.calculate_fbeta_score(y_true, y_pred, threshold)
            metrics[f'fbeta_thresh_{threshold}'] = fbeta
        
        best_threshold = max(self.thresholds, key=lambda t: self.calculate_fbeta_score(y_true, y_pred, t))
        metrics['best_fbeta'] = self.calculate_fbeta_score(y_true, y_pred, best_threshold)
        metrics['best_threshold'] = best_threshold
        
        return metrics


class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics_calculator = MetricsCalculator(config)
        
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get_log_path(0), delay=True),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_optimizer(self, model: ResNextClassifier, stage: int = 1) -> torch.optim.Optimizer:
        if stage == 1:
            lr = self.config.learning_rate
        else:
            lr = self.config.learning_rate * 0.1
        
        return Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, 
                        stage: int = 1) -> torch.optim.lr_scheduler._LRScheduler:
        if stage == 1:
            return CosineAnnealingLR(optimizer, T_max=1, eta_min=self.config.learning_rate)
        else:
            return CosineAnnealingLR(optimizer, T_max=self.config.epochs, 
                                    eta_min=self.config.learning_rate * 0.01)
    
    def train_epoch(self, model: ResNextClassifier, train_loader, 
                   optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> Dict[str, float]:
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            # BatchNorm cannot estimate variance from a singleton spatial cell.
            if images.size(0) == 1:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            batch_size = outputs.size(0)
            (batch_size * loss).backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                predictions = torch.sigmoid(outputs).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.append(predictions)
                all_labels.append(labels_np)
            
            model.train()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate_epoch(self, model: ResNextClassifier, valid_loader, 
                      loss_fn: nn.Module) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                total_loss += loss.item()
                
                predictions = torch.sigmoid(outputs).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.append(predictions)
                all_labels.append(labels_np)
        
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(valid_loader)
        
        return metrics
    
    def train_stage(self, model: ResNextClassifier, train_loader, valid_loader,
                   stage: int, epochs: int, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        
        self.logger.info(f"Starting Stage {stage} training...")
        
        optimizer = self.create_optimizer(model, stage)
        scheduler = self.create_scheduler(optimizer, stage)
        loss_fn = ModelFactory.create_loss_function('focal', self.config)
        
        best_metric = -float("inf")
        best_metrics = None
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(model, train_loader, optimizer, loss_fn)
            val_metrics = self.validate_epoch(model, valid_loader, loss_fn)
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train F-Beta: {train_metrics['best_fbeta']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F-Beta: {val_metrics['best_fbeta']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            if val_metrics['best_fbeta'] > best_metric:
                best_metric = val_metrics['best_fbeta']
                patience_counter = 0
                
                best_metrics = dict(val_metrics)
                if checkpoint_path is not None:
                    save_model_checkpoint(
                        model, epoch, val_metrics['loss'],
                        val_metrics['best_fbeta'], checkpoint_path
                    )
                self.logger.info(f"New best model saved with F-Beta: {best_metric:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if best_metrics is None:
            raise ValueError('Training must run at least one epoch with finite metrics')
        return best_metrics
    
    def train_fold(self, fold_idx: int) -> Dict[str, float]:
        self.logger.info(f"Training fold {fold_idx}...")
        
        random.seed(self.config.seed + fold_idx)
        np.random.seed(self.config.seed + fold_idx)
        torch.manual_seed(self.config.seed + fold_idx)
        train_loader, valid_loader = create_data_loaders(self.config, fold_idx)
        
        model = ModelFactory.create_model(self.config).to(self.device)
        
        stage1_path = self.config.get_model_path(fold_idx, 'stage_1')
        stage2_path = self.config.get_model_path(fold_idx, 'stage_2')
        
        self.logger.info("Stage 1: Training with frozen backbone...")
        model.freeze()
        self.train_stage(
            model, train_loader, valid_loader, 
            stage=1, epochs=1, checkpoint_path=stage1_path
        )
        
        self.logger.info("Stage 2: Fine-tuning entire model...")
        load_model_checkpoint(model, stage1_path)
        model.unfreeze_backbone()
        stage2_metrics = self.train_stage(
            model, train_loader, valid_loader,
            stage=2, epochs=self.config.epochs, checkpoint_path=stage2_path
        )
        
        self.logger.info(f"Fold {fold_idx} completed. Best F-Beta: {stage2_metrics['best_fbeta']:.4f}")
        
        del model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return stage2_metrics
    
    def train_all_folds(self) -> Dict[int, Dict[str, float]]:
        results = {}
        
        folds_to_train = [self.config.fold_idx] if self.config.fold_idx is not None else range(1, self.config.num_folds + 1)
        
        for fold_idx in folds_to_train:
            fold_results = self.train_fold(fold_idx)
            results[fold_idx] = fold_results
        
        self.logger.info("All folds completed!")
        return results
