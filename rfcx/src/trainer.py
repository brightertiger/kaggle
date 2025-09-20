import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from typing import Tuple, Dict, Any
from .config import Config

class MetricsCalculator:
    @staticmethod
    def calculate_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
        pred_classes = predictions.argmax(axis=1)
        true_classes = labels.argmax(axis=1)
        accuracy = (pred_classes == true_classes).astype(float).mean()
        return round(accuracy, 4)

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics_calculator = MetricsCalculator()

    def validate_model(self, model: nn.Module, valid_loader: torch.utils.data.DataLoader, 
                      loss_fn: nn.Module) -> Tuple[float, float]:
        model.eval()
        losses = []
        label_array = []
        pred_array = []
        
        with torch.no_grad():
            for sample in valid_loader:
                image = Variable(sample[0].float().to(self.device))
                label = Variable(sample[1].float().squeeze().to(self.device))
                
                preds = model(image).squeeze()
                loss = loss_fn(preds, label).sum()
                
                label_array.append(label.cpu().data.numpy())
                pred_array.append(preds.cpu().data.numpy())
                losses.append(loss.data.item())
        
        label_array = np.vstack(label_array)
        pred_array = np.vstack(pred_array)
        
        mean_loss = round(np.mean(losses), 4)
        mean_accuracy = self.metrics_calculator.calculate_accuracy(label_array, pred_array)
        
        model.train()
        return mean_loss, mean_accuracy

    def save_model(self, epoch: int, model: nn.Module, metric: float, path: str) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric': metric
        }
        torch.save(checkpoint, path)

    def train_fold(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                   valid_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
                   loss_fn: nn.Module, fold: int) -> None:
        
        best_metric = -100.0
        counter = 0
        
        log_file = os.path.join(self.config.data.model_save_path, f'log_fold_{fold}.txt')
        if os.path.exists(log_file):
            os.remove(log_file)
        
        log_file = open(log_file, 'w', buffering=1)
        
        for epoch in range(self.config.training.epochs):
            model.train()
            tq = tqdm(total=len(train_loader) * self.config.training.batch_size, 
                     ncols=0, disable=False)
            
            losses = []
            label_array = []
            pred_array = []
            
            optimizer.zero_grad()
            
            for sample in train_loader:
                image = Variable(sample[0].float().to(self.device))
                label = Variable(sample[1].float().squeeze().to(self.device))
                
                preds = model(image).squeeze()
                loss = loss_fn(preds, label)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                losses.append(loss.data.item())
                train_loss = round(np.mean(losses), 4)
                
                tq.update(self.config.training.batch_size)
                tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
                
                label_array.append(label.cpu().data.numpy())
                pred_array.append(preds.cpu().data.numpy())
            
            label_array = np.vstack(label_array)
            pred_array = np.vstack(pred_array)
            
            valid_loss, valid_metric = self.validate_model(model, valid_loader, loss_fn)
            
            postfix = OrderedDict()
            postfix['trn_ls'] = '{:.4f}'.format(train_loss)
            postfix['val_ls'] = '{:.4f}'.format(valid_loss)
            postfix['val_mt'] = '{:.4f}'.format(valid_metric)
            tq.set_postfix(**postfix)
            tq.close()
            
            if valid_metric > best_metric:
                counter = 0
                best_metric = valid_metric
                save_path = os.path.join(self.config.data.model_save_path, f'model_fold_{fold}.pt')
                self.save_model(epoch, model, valid_metric, save_path)
            else:
                counter += 1
            
            log_text = f'Epoch - {epoch} | '
            log_text += f'Train Loss - {train_loss:.4f} | '
            log_text += f'Valid Loss - {valid_loss:.4f} | '
            log_text += f'Valid Metric - {valid_metric:.4f} | \n'
            log_file.write(log_text)
            
            scheduler.step(-1. * valid_metric)
        
        log_file.close()

def train_model(config: Config, model_type: str = "resnet") -> None:
    from .models import create_model
    from .data_utils import create_data_loaders
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch.nn as nn
    
    trainer = ModelTrainer(config)
    
    for fold in range(1, config.training.num_folds + 1):
        print(f"\nTraining fold {fold}...")
        
        train_loader, valid_loader = create_data_loaders(config, fold)
        
        model = create_model(config, model_type)
        model = model.to(config.device)
        
        optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            min_lr=config.training.min_lr, 
            patience=config.training.patience, 
            factor=config.training.factor
        )
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        trainer.train_fold(model, train_loader, valid_loader, optimizer, 
                          scheduler, loss_fn, fold)
        
        model.cpu()
        del model
        torch.cuda.empty_cache()
