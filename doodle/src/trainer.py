import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from prettytable import PrettyTable

from .models import ResNetClassifier, TopKAccuracy
from .data_utils import create_dataloaders
from .config import Config


class ModelTrainer:
    def __init__(self, 
                 config: Config,
                 model_name: str = 'resnet50',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        
        self.config = config
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.model_path = os.path.join(config.model_path, model_name)
        os.makedirs(self.model_path, exist_ok=True)
        
        self.category_mapping = self._load_categories()
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = TopKAccuracy(k=3)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.train_loader, self.valid_loader = None, None
        self.best_metric = 0.0
        self.patience_counter = 0

    def _load_categories(self) -> list:
        categories_path = os.path.join(self.config.data_path, 'categories.pkl')
        with open(categories_path, 'rb') as f:
            categories = pickle.load(f)
        return [cat.replace('.csv', '') for cat in categories]

    def _create_model(self) -> nn.Module:
        model = ResNetClassifier(
            model_name=self.model_name,
            num_classes=self.config.num_classes
        )
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        return model.to(self.config.device)

    def prepare_data(self, 
                    train_df: pd.DataFrame,
                    valid_df: pd.DataFrame):
        self.train_loader, self.valid_loader = create_dataloaders(
            train_df, valid_df, self.category_mapping, self.config
        )

    def save_checkpoint(self, filepath: str):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'model_name': self.model_name
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        state = torch.load(filepath, map_location=self.config.device)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_metric = state.get('best_metric', 0.0)
        print(f"Model loaded from {filepath}")

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        train_losses = []
        train_metrics = []
        
        for batch_idx, sample in enumerate(self.train_loader):
            images = sample['image'].to(self.config.device)
            labels = sample['label'].to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            metric = self.accuracy_metric(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            train_metrics.append(metric.item())
        
        return np.mean(train_losses), np.mean(train_metrics)

    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        valid_losses = []
        valid_metrics = []
        
        with torch.no_grad():
            for sample in self.valid_loader:
                images = sample['image'].to(self.config.device)
                labels = sample['label'].to(self.config.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                metric = self.accuracy_metric(outputs, labels)
                
                valid_losses.append(loss.item())
                valid_metrics.append(metric.item())
        
        return np.mean(valid_losses), np.mean(valid_metrics)

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
        self.prepare_data(train_df, valid_df)
        
        log_table = PrettyTable(['Phase', 'Epoch', 'LR', 'Loss', 'Top-3 Acc'])
        log_file = os.path.join(self.model_path, f'{self.model_name}_training.log')
        
        for epoch in range(self.config.epochs):
            train_loss, train_metric = self.train_epoch()
            valid_loss, valid_metric = self.validate_epoch()
            
            log_table.add_row(['Train', epoch, self.learning_rate, 
                             round(train_loss, 4), round(train_metric, 4)])
            log_table.add_row(['Valid', epoch, self.learning_rate, 
                             round(valid_loss, 4), round(valid_metric, 4)])
            
            with open(log_file, 'w') as f:
                f.write(log_table.get_string())
            
            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                checkpoint_path = os.path.join(self.model_path, f'{self.model_name}_best.pth')
                self.save_checkpoint(checkpoint_path)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.patience:
                self.learning_rate *= 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                self.patience_counter = 0
            
            if self.learning_rate < 1e-7:
                print("Learning rate too small, stopping training")
                break
        
        return {
            'best_metric': self.best_metric,
            'final_lr': self.learning_rate,
            'epochs_trained': epoch + 1
        }
