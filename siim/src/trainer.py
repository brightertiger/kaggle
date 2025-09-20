import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from apex import amp
from .config import Config

class MelanomaTrainer:
    def __init__(self, model, train_loader, valid_loader, device=Config.DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.8, 
            patience=2, 
            min_lr=1e-8,
            verbose=True
        )
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device)
        )
        
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level='O2', 
            verbosity=False
        )
        
        self.best_score = 0.0
        self.train_losses = []
        self.valid_losses = []
        self.valid_scores = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            targets = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, metadata)
            loss = self.criterion(outputs, targets)
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            
            if (batch_idx + 1) % 2 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            predictions = torch.sigmoid(outputs).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets_np)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, all_predictions, all_targets
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                targets = batch['label'].to(self.device)
                
                outputs = self.model(images, metadata)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                predictions = torch.sigmoid(outputs).cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets_np)
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        avg_loss = total_loss / len(self.valid_loader)
        
        # Calculate AUC for melanoma class (class 1)
        melanoma_predictions = all_predictions[:, 1]
        melanoma_targets = all_targets[:, 1]
        auc_score = roc_auc_score(melanoma_targets, melanoma_predictions)
        
        return avg_loss, auc_score, all_predictions, all_targets
    
    def train(self, epochs=Config.NUM_EPOCHS, save_path=None):
        if save_path is None:
            save_path = Config.MODEL_DIR / 'melanoma_model.pt'
        
        os.makedirs(save_path.parent, exist_ok=True)
        
        log_file = save_path.parent / 'training_log.txt'
        with open(log_file, 'w') as f:
            f.write('Epoch,Train_Loss,Valid_Loss,Valid_AUC,LR\n')
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)
            
            # Training
            train_loss, train_preds, train_targets = self.train_epoch()
            
            # Validation
            valid_loss, valid_auc, valid_preds, valid_targets = self.validate()
            
            # Update learning rate
            self.scheduler.step(valid_auc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.valid_scores.append(valid_auc)
            
            # Save best model
            if valid_auc > self.best_score:
                self.best_score = valid_auc
                self.save_model(save_path, epoch, valid_auc)
                print(f'New best model saved! AUC: {valid_auc:.4f}')
            
            # Log results
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
            print(f'Valid AUC: {valid_auc:.4f}')
            print(f'Learning Rate: {current_lr:.2e}')
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f'{epoch},{train_loss:.4f},{valid_loss:.4f},{valid_auc:.4f},{current_lr:.2e}\n')
        
        print(f'\nTraining completed! Best AUC: {self.best_score:.4f}')
        return self.best_score
    
    def save_model(self, path, epoch, score):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': score,
            'config': Config.__dict__
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score = checkpoint['best_score']
        return checkpoint['epoch']
