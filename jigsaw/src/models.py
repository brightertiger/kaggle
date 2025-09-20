#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from .config import Config


class ToxicCommentDataset(Dataset):
    """Dataset class for toxic comment classification."""
    
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 222):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aux_labels = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    
    def __len__(self):
        return len(self.data)
    
    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text and return token IDs."""
        if hasattr(self.tokenizer, 'tokenize'):
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens[:self.max_length-2] + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            token_ids = encoding['input_ids'].flatten()
        
        token_ids = np.array(token_ids)
        if len(token_ids) < self.max_length:
            token_ids = np.pad(token_ids, (0, self.max_length - len(token_ids)), 'constant')
        
        return token_ids
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text = str(row['comment_text'])
        token_ids = self._tokenize_text(text)
        
        weight = float(row['weight'])
        target = float(row['target'])
        
        aux_labels = np.array([float(row[label]) for label in self.aux_labels])
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'weight': torch.tensor(weight, dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.float),
            'aux_labels': torch.tensor(aux_labels, dtype=torch.float),
            'id': row['id']
        }


class BERTClassifier(nn.Module):
    """BERT-based classifier for toxic comment classification."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.bert_config = config.bert_config
        
        try:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(self.bert_config['model_name'])
        except ImportError:
            from pytorch_pretrained_bert.modeling import BertModel
            self.bert = BertModel.from_pretrained(self.bert_config['model_name'])
        
        self.dropout = nn.Dropout(self.bert_config['dropout'])
        self.classifier = nn.Linear(self.bert_config['hidden_size'], 1)
        self.aux_classifier = nn.Linear(self.bert_config['hidden_size'], len(config.aux_labels))
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        attention_mask = (input_ids > 0).long()
        
        if hasattr(self.bert, 'forward'):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output'):
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs[1]
        else:
            _, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        
        pooled_output = self.dropout(pooled_output)
        
        main_output = self.classifier(pooled_output)
        aux_output = self.aux_classifier(pooled_output)
        
        return main_output, aux_output


class GPTClassifier(nn.Module):
    """GPT-2-based classifier for toxic comment classification."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gpt_config = config.gpt_config
        
        try:
            from transformers import GPT2Model
            self.transformer = GPT2Model.from_pretrained(self.gpt_config['model_name'])
        except ImportError:
            from pytorch_pretrained_bert import GPT2Model
            self.transformer = GPT2Model.from_pretrained(self.gpt_config['model_name'])
        
        self.dropout = nn.Dropout(self.gpt_config['dropout'])
        self.classifier = nn.Linear(self.gpt_config['hidden_size'] * 2, 1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        transformer_outputs = self.transformer(input_ids)
        
        if hasattr(transformer_outputs, 'last_hidden_state'):
            hidden_states = transformer_outputs.last_hidden_state
        else:
            hidden_states = transformer_outputs[0]
        
        avg_pool = torch.mean(hidden_states, dim=1)
        max_pool, _ = torch.max(hidden_states, dim=1)
        
        concatenated = torch.cat([avg_pool, max_pool], dim=1)
        concatenated = self.dropout(concatenated)
        
        output = self.classifier(concatenated)
        return output


class CustomLoss:
    """Custom loss functions for toxic comment classification."""
    
    @staticmethod
    def bert_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                 aux_predictions: torch.Tensor, aux_targets: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
        """Custom loss for BERT model with auxiliary tasks."""
        main_loss = F.mse_loss(predictions.squeeze(), targets.squeeze(), reduction='none')
        weighted_main_loss = torch.mean(weights * main_loss)
        
        aux_loss = F.mse_loss(aux_predictions, aux_targets)
        
        total_loss = 3.5 * weighted_main_loss + aux_loss
        return total_loss
    
    @staticmethod
    def gpt_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                weights: torch.Tensor) -> torch.Tensor:
        """Custom loss for GPT model."""
        loss = F.binary_cross_entropy_with_logits(
            predictions.squeeze(), targets.squeeze(), weight=weights
        )
        return loss


class ModelTrainer:
    """Model training utilities."""
    
    def __init__(self, config: Config, model_type: str = 'bert'):
        self.config = config
        self.model_type = model_type
        self.model_config = config.bert_config if model_type == 'bert' else config.gpt_config
        self.training_config = config.training_config
    
    def create_data_loaders(self, train_data: pd.DataFrame, valid_data: pd.DataFrame,
                          tokenizer) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation."""
        train_dataset = ToxicCommentDataset(train_data, tokenizer, self.config.max_length)
        valid_dataset = ToxicCommentDataset(valid_data, tokenizer, self.config.max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config['num_workers'],
            pin_memory=self.training_config['pin_memory'],
            drop_last=self.training_config['drop_last']
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.model_config['valid_batch_size'],
            shuffle=False,
            num_workers=self.training_config['num_workers'],
            pin_memory=self.training_config['pin_memory'],
            drop_last=False
        )
        
        return train_loader, valid_loader
    
    def setup_optimizer(self, model: nn.Module, num_training_steps: int):
        """Setup optimizer with proper parameter groups."""
        if self.model_type == 'bert':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        else:
            no_decay = ['bias', '.ln']
        
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.model_config['weight_decay']
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        try:
            from transformers import AdamW
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_config['learning_rate'])
        except ImportError:
            from pytorch_pretrained_bert import BertAdam
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=self.model_config['learning_rate'],
                warmup=self.model_config['warmup_ratio'],
                t_total=num_training_steps
            )
        
        return optimizer
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer, loss_fn, device: str) -> float:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            weights = batch['weight'].to(device)
            targets = batch['target'].to(device)
            aux_targets = batch['aux_labels'].to(device)
            
            optimizer.zero_grad()
            
            if self.model_type == 'bert':
                predictions, aux_predictions = model(input_ids)
                loss = loss_fn(predictions, targets, aux_predictions, aux_targets, weights)
            else:
                predictions = model(input_ids)
                loss = loss_fn(predictions, targets, weights)
            
            loss.backward()
            
            if self.model_config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_config['max_grad_norm'])
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_model(self, model: nn.Module, valid_loader: DataLoader, 
                      device: str) -> Tuple[float, pd.DataFrame]:
        """Validate model and return predictions."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_ids = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                weights = batch['weight'].to(device)
                targets = batch['target'].to(device)
                ids = batch['id']
                
                if self.model_type == 'bert':
                    predictions, _ = model(input_ids)
                else:
                    predictions = model(input_ids)
                
                predictions = torch.sigmoid(predictions).cpu().numpy()
                all_predictions.extend(predictions.flatten())
                all_ids.extend(ids.numpy())
        
        predictions_df = pd.DataFrame({
            'id': all_ids,
            'prediction': all_predictions
        })
        
        return total_loss, predictions_df
    
    def save_model(self, model: nn.Module, optimizer, epoch: int, 
                  loss: float, save_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, model: nn.Module, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {checkpoint_path}")
        return model
