import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, log_loss
from ..utils.config import Config
from ..models.loss import reduce_loss

class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer, 
                 scheduler, save_path, subset):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.subset = subset
        self.device = Config.DEVICE
        
    def save_model(self, epoch, loss, metric):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'metric': metric,
            'epoch': epoch
        }
        torch.save(checkpoint, f'{self.save_path}/model_{self.subset}.pt')
    
    def validate(self):
        self.model.eval()
        scores = []
        labels = []
        
        with torch.no_grad():
            for batch in self.valid_loader:
                batch_labels = batch.pop('label').reshape(-1, 1)
                batch_labels = batch_labels.to(self.device).squeeze().float()
                
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                predictions = torch.sigmoid(self.model(**batch))
                scores.append(predictions.cpu().data.numpy())
                labels.append(batch_labels.cpu().data.numpy())
                
                for key, value in batch.items():
                    batch[key] = value.to('cpu')
        
        torch.cuda.empty_cache()
        
        scores = np.vstack(scores).reshape(-1, 1)
        labels = np.vstack(labels).reshape(-1, 1)
        
        loss = round(log_loss(labels, scores), 4)
        metric = round(roc_auc_score(labels, scores), 4)
        
        self.model.train()
        return loss, metric
    
    def train_epoch(self, epoch):
        step = 0
        self.optimizer.zero_grad()
        self.train_loader.dataset.epoch(epoch)
        
        losses = []
        tq = tqdm(total=len(self.train_loader) * Config.BATCH_SIZE, disable=False)
        
        for batch in self.train_loader:
            step += 1
            
            batch_labels = batch.pop('label').squeeze().float()
            batch_labels = batch_labels.to(self.device).reshape(-1, 1)
            
            batch_weights = batch.pop('weight').squeeze().float()
            batch_weights = batch_weights.to(self.device).reshape(-1, 1)
            
            for key, value in batch.items():
                batch[key] = value.to(self.device)
            
            predictions = self.model(**batch)
            batch_size = predictions.shape[0]
            
            loss = reduce_loss(self.loss_fn(predictions, batch_labels, batch_weights))
            loss.backward()
            
            if step % 4 == 0:
                step = 0
                clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            losses.append(loss.cpu().data.item())
            train_loss = round(np.mean(losses), 4)
            
            tq.update(batch_size)
            tq.set_postfix(train_loss=f'{train_loss:.4f}')
            
            for key, value in batch.items():
                batch[key] = value.to('cpu')
        
        del batch, batch_labels, predictions
        torch.cuda.empty_cache()
        tq.close()
        
        return train_loss
    
    def train(self, epochs):
        logfile_path = f'{self.save_path}/logfile_{self.subset}.txt'
        if os.path.exists(logfile_path):
            os.remove(logfile_path)
        
        logfile = open(logfile_path, 'w', buffering=1)
        self.model.train()
        
        best_metric = float('-inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss, valid_metric = self.validate()
            
            self.scheduler.step(valid_loss)
            
            if valid_metric > best_metric:
                self.save_model(epoch, valid_loss, valid_metric)
                best_metric = valid_metric
            
            log_text = f'Epoch - {epoch} | '
            log_text += f'Train Loss - {train_loss:.4f} | '
            log_text += f'Valid Loss - {valid_loss:.4f} | '
            log_text += f'Valid Metric - {valid_metric:.4f} | \n'
            
            logfile.write(log_text)
            logfile.flush()
        
        logfile.close()
