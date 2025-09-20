import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import cohen_kappa_score
from apex import amp
from config import Config

class DiabeticRetinopathyTrainer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")
    
    def calculate_quadratic_kappa(self, true_labels: np.ndarray, predictions: np.ndarray) -> float:
        true_labels = np.rint(true_labels)
        predictions = np.rint(predictions.clip(0., 4.))
        return cohen_kappa_score(true_labels, predictions, weights='quadratic')
    
    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.sum() / loss.shape[0]
    
    def save_model(self, epoch: int, model: torch.nn.Module, loss: float, path: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, path)
    
    def validate_model(self, model: torch.nn.Module, data_loader, loss_fn) -> tuple:
        model.eval()
        losses = []
        true_labels = []
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].float().to(self.device)
                labels = batch['label'].float().squeeze().to(self.device)
                weights = batch['weight'].float().squeeze().to(self.device)
                
                regression_output, classification_output = model(images)
                loss = self.reduce_loss(loss_fn(regression_output, classification_output, labels, weights))
                
                losses.append(loss.item())
                true_labels.append(labels.cpu().numpy())
                predictions.append(regression_output.cpu().numpy())
        
        true_labels = np.hstack(true_labels)
        predictions = np.hstack(predictions)
        kappa_score = self.calculate_quadratic_kappa(true_labels, predictions)
        
        model.train()
        return np.mean(losses), kappa_score
    
    def train_epoch(self, model: torch.nn.Module, data_loader, loss_fn, optimizer) -> tuple:
        model.train()
        losses = []
        true_labels = []
        predictions = []
        
        progress_bar = tqdm(total=len(data_loader) * self.config.BATCH_SIZE, 
                           ncols=0, disable=False)
        
        for step, batch in enumerate(data_loader):
            images = batch['image'].float().to(self.device)
            labels = batch['label'].float().squeeze().to(self.device)
            weights = batch['weight'].float().squeeze().to(self.device)
            
            regression_output, classification_output = model(images)
            loss = self.reduce_loss(loss_fn(regression_output, classification_output, labels, weights))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                (self.config.BATCH_SIZE * scaled_loss).backward()
            
            if step % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())
            true_labels.append(labels.cpu().numpy())
            predictions.append(regression_output.cpu().numpy())
            
            progress_bar.update(self.config.BATCH_SIZE)
            progress_bar.set_postfix(trn_ls=f'{np.mean(losses):.5f}')
        
        true_labels = np.hstack(true_labels)
        predictions = np.hstack(predictions)
        kappa_score = self.calculate_quadratic_kappa(true_labels, predictions)
        
        progress_bar.close()
        return np.mean(losses), kappa_score
    
    def train_model(self, model: torch.nn.Module, train_loader, valid_loader, 
                   loss_fn, optimizer, scheduler, save_path: str, epochs: int) -> None:
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_kappa = self.train_epoch(model, train_loader, loss_fn, optimizer)
            valid_loss, valid_kappa = self.validate_model(model, valid_loader, loss_fn)
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.5f}, Train Kappa: {train_kappa:.5f}')
            print(f'  Valid Loss: {valid_loss:.5f}, Valid Kappa: {valid_kappa:.5f}')
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
                self.save_model(epoch, model, valid_loss, save_path)
                print(f'  New best model saved!')
            else:
                patience_counter += 1
                print(f'  No improvement ({patience_counter} epochs)')

class NoiseAugmentedTrainer(DiabeticRetinopathyTrainer):
    def validate_model(self, model: torch.nn.Module, data_loader, loss_fn) -> tuple:
        model.eval()
        losses = []
        true_labels = []
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                images_1 = batch['image_1'].float().to(self.device)
                images_2 = batch['image_2'].float().to(self.device)
                labels = batch['label'].float().squeeze().to(self.device)
                weights = batch['weight'].float().squeeze().to(self.device)
                
                reg_1, cls_1 = model(images_1)
                reg_2, cls_2 = model(images_2)
                
                loss = self.reduce_loss(loss_fn(reg_1, reg_2, cls_1, cls_2, labels, weights))
                
                losses.append(loss.item())
                true_labels.append(labels.cpu().numpy())
                predictions.append(((reg_1 + reg_2) / 2).cpu().numpy())
        
        true_labels = np.hstack(true_labels)
        predictions = np.hstack(predictions)
        kappa_score = self.calculate_quadratic_kappa(true_labels, predictions)
        
        model.train()
        return np.mean(losses), kappa_score
    
    def train_epoch(self, model: torch.nn.Module, data_loader, loss_fn, optimizer) -> tuple:
        model.train()
        losses = []
        true_labels = []
        predictions = []
        
        progress_bar = tqdm(total=len(data_loader) * self.config.BATCH_SIZE, 
                           ncols=0, disable=False)
        
        for step, batch in enumerate(data_loader):
            images_1 = batch['image_1'].float().to(self.device)
            images_2 = batch['image_2'].float().to(self.device)
            labels = batch['label'].float().squeeze().to(self.device)
            weights = batch['weight'].float().squeeze().to(self.device)
            
            reg_1, cls_1 = model(images_1)
            reg_2, cls_2 = model(images_2)
            
            loss = self.reduce_loss(loss_fn(reg_1, reg_2, cls_1, cls_2, labels, weights))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                (self.config.BATCH_SIZE * scaled_loss).backward()
            
            if step % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item())
            true_labels.append(labels.cpu().numpy())
            predictions.append(((reg_1 + reg_2) / 2).cpu().numpy())
            
            progress_bar.update(self.config.BATCH_SIZE)
            progress_bar.set_postfix(trn_ls=f'{np.mean(losses):.5f}')
        
        true_labels = np.hstack(true_labels)
        predictions = np.hstack(predictions)
        kappa_score = self.calculate_quadratic_kappa(true_labels, predictions)
        
        progress_bar.close()
        return np.mean(losses), kappa_score
