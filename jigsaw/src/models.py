"""BERT multitask regression and GPT-2 toxicity classification."""
import os
from typing import Tuple

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model
from transformers import get_linear_schedule_with_warmup

from .config import Config


class ToxicCommentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=222, aux_labels=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aux_labels = aux_labels or Config().aux_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = 'none blank' if pd.isna(row['comment_text']) else str(row['comment_text'])
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        result = {key: encoding[key].squeeze(0) for key in ('input_ids', 'attention_mask')}
        result['id'] = row['id']
        if 'target' in self.data.columns:
            result.update(
                weight=torch.tensor(float(row['weight']), dtype=torch.float32),
                target=torch.tensor(float(row['target']), dtype=torch.float32),
                aux_labels=torch.tensor([float(row[c]) for c in self.aux_labels]),
            )
        return result


class BERTClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        settings = config.bert_config
        if config.random_init:
            self.bert = BertModel(BertConfig(**settings['backbone_config']))
        else:
            self.bert = BertModel.from_pretrained(settings['model_name'])
        self.dropout = nn.Dropout(settings['dropout'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.aux_classifier = nn.Linear(self.bert.config.hidden_size, len(config.aux_labels))

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.bert.config.pad_token_id).long()
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled), self.aux_classifier(pooled)


class GPTClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        settings = config.gpt_config
        if config.random_init:
            self.transformer = GPT2Model(GPT2Config(**settings['backbone_config']))
        else:
            self.transformer = GPT2Model.from_pretrained(settings['model_name'])
        self.dropout = nn.Dropout(settings['dropout'])
        self.classifier = nn.Linear(self.transformer.config.hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            pad_id = self.transformer.config.pad_token_id
            if pad_id is None:
                pad_id = self.transformer.config.eos_token_id
            attention_mask = input_ids.ne(pad_id).long()
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).bool()
        avg_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        max_pool = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min).max(dim=1).values
        max_pool = torch.where(mask.any(dim=1), max_pool, torch.zeros_like(max_pool))
        return self.classifier(self.dropout(torch.cat([avg_pool, max_pool], dim=1)))


def prediction_scores(logits, model_type):
    # Retain the original monotonic score mapping for both branches.
    # BERT uses MSE, so its sigmoid scores should not be read as calibrated probabilities.
    return logits.reshape(-1).sigmoid()


class CustomLoss:
    @staticmethod
    def bert_loss(predictions, targets, aux_predictions, aux_targets, weights):
        main_loss = F.mse_loss(predictions.reshape(-1), targets.reshape(-1), reduction='none')
        return 3.5 * (weights.reshape(-1) * main_loss).mean() + F.mse_loss(aux_predictions, aux_targets)

    @staticmethod
    def gpt_loss(predictions, targets, weights):
        return F.binary_cross_entropy_with_logits(
            predictions.reshape(-1), targets.reshape(-1), weight=weights.reshape(-1)
        )


class ModelTrainer:
    def __init__(self, config: Config, model_type='bert'):
        if model_type not in ('bert', 'gpt'):
            raise ValueError(f'Unknown model type: {model_type}')
        self.config = config
        self.model_type = model_type
        self.model_config = config.bert_config if model_type == 'bert' else config.gpt_config
        self.training_config = config.training_config
        self.scheduler = None
        self.scaler = None

    def create_data_loaders(self, train_data, valid_data, tokenizer) -> Tuple[DataLoader, DataLoader]:
        common = dict(num_workers=self.training_config['num_workers'],
                      pin_memory=self.training_config['pin_memory'] and torch.cuda.is_available())
        train_loader = DataLoader(
            ToxicCommentDataset(train_data, tokenizer, self.config.max_length, self.config.aux_labels),
            batch_size=self.model_config['batch_size'], shuffle=True,
            drop_last=self.training_config['drop_last'], **common,
        )
        valid_loader = DataLoader(
            ToxicCommentDataset(valid_data, tokenizer, self.config.max_length, self.config.aux_labels),
            batch_size=self.model_config['valid_batch_size'], shuffle=False, **common,
        )
        if not len(train_loader) or not len(valid_loader):
            raise ValueError('Empty loader: add rows, reduce batch size, or disable drop_last.')
        return train_loader, valid_loader

    def setup_optimizer(self, model, num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight'] if self.model_type == 'bert' else ['bias', '.ln']
        parameters = list(model.named_parameters())
        groups = [
            {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': self.model_config['weight_decay']},
            {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=self.model_config['learning_rate'])
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer, int(num_training_steps * self.model_config['warmup_ratio']), num_training_steps
        )
        return optimizer

    def _batch_loss(self, model, batch, device, loss_fn):
        outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        target, weight = batch['target'].to(device), batch['weight'].to(device)
        if self.model_type == 'bert':
            logits, auxiliary = outputs
            loss = loss_fn(logits, target, auxiliary, batch['aux_labels'].to(device), weight)
        else:
            logits = outputs
            loss = loss_fn(logits, target, weight)
        return loss, logits

    def train_epoch(self, model, train_loader, optimizer, loss_fn, device):
        model.train()
        accumulation = self.model_config['gradient_accumulation_steps']
        if accumulation < 1:
            raise ValueError('gradient_accumulation_steps must be positive')
        use_amp = self.training_config['mixed_precision'] and torch.device(device).type == 'cuda'
        if self.scaler is None:
            self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_samples = 0
        for index, batch in enumerate(tqdm(train_loader, desc='Training')):
            # Normalize the final, potentially incomplete accumulation group too.
            group_start = index // accumulation * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
                loss, _ = self._batch_loss(model, batch, device, loss_fn)
            self.scaler.scale(loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                self.scaler.unscale_(optimizer)
                if self.model_config['max_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.model_config['max_grad_norm'])
                previous_scale = self.scaler.get_scale()
                self.scaler.step(optimizer)
                self.scaler.update()
                if self.scheduler is not None and self.scaler.get_scale() >= previous_scale:
                    self.scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            count = len(batch['target'])
            total_loss += loss.item() * count
            total_samples += count
        return total_loss / total_samples

    def validate_model(self, model, valid_loader, device):
        model.eval()
        loss_fn = CustomLoss.bert_loss if self.model_type == 'bert' else CustomLoss.gpt_loss
        total_loss, total_samples = 0.0, 0
        predictions, ids = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Validation'):
                loss, logits = self._batch_loss(model, batch, device, loss_fn)
                count = len(batch['target'])
                total_loss += loss.item() * count
                total_samples += count
                predictions.extend(prediction_scores(logits, self.model_type).cpu().tolist())
                ids.extend(batch['id'].tolist() if torch.is_tensor(batch['id']) else batch['id'])
        return total_loss / total_samples, pd.DataFrame({'id': ids, 'prediction': predictions})

    def save_model(self, model, optimizer, epoch, loss, save_path):
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss,
            'model_config': self.model_config, 'training_config': self.training_config,
            'config': self.config.to_dict(),
        }, save_path)
        print(f'Model saved to: {save_path}')

    def load_model(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
