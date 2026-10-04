"""Fold training, gradient accumulation, validation, and portable checkpoints."""
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .config import Config
from .models import TweetSentimentModel, TweetLoss


def resolve_device(config: Config):
    requested = config.training.device
    if requested.startswith('cuda') and not torch.cuda.is_available():
        requested = 'cpu'
    return torch.device(requested)


class TweetTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = resolve_device(config)
        self.use_amp = config.training.mixed_precision and self.device.type == 'cuda'

    def _setup_optimizer(self, model: nn.Module):
        excluded = ['bias', 'LayerNorm.weight']
        params = list(model.named_parameters())
        groups = [
            {'params': [p for n, p in params if not any(ex in n for ex in excluded)],
             'weight_decay': self.config.model.weight_decay},
            {'params': [p for n, p in params if any(ex in n for ex in excluded)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(groups, lr=self.config.model.learning_rate, eps=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, factor=self.config.model.scheduler_factor,
                                      min_lr=self.config.model.scheduler_min_lr,
                                      patience=self.config.model.scheduler_patience)
        return optimizer, scheduler

    def _save_checkpoint(self, model, fold, epoch, loss):
        path = Path(self.config.data.model_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(), 'loss': loss, 'epoch': epoch,
            'config': self.config.to_dict(), 'encoder_config': model.encoder.config.to_dict(),
        }, path / f'model_fold_{fold}.pt')

    def _batch_loss(self, model, batch, loss_fn):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            start, end, auxiliary = model(batch['tokens'], batch['masks'], batch['text_mask'])
            loss = loss_fn(start, end, batch['start_idx'], batch['end_idx'], auxiliary, batch['aux_label'])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite span loss; check labels and token masks')
        return loss

    def _validate(self, model, valid_loader, loss_fn):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                loss = self._batch_loss(model, batch, loss_fn)
                size = len(batch['tokens'])
                total += loss.item() * size
                count += size
        if not count:
            raise ValueError('Validation loader is empty')
        return total / count

    def train_fold(self, fold, train_loader, valid_loader):
        if not len(train_loader) or self.config.model.max_epochs < 1:
            raise ValueError('Training requires a nonempty loader and at least one epoch')
        accumulation = self.config.model.gradient_accumulation_steps
        if accumulation < 1:
            raise ValueError('gradient_accumulation_steps must be positive')
        seed = self.config.data.random_seed + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = TweetSentimentModel(self.config).to(self.device)
        optimizer, scheduler = self._setup_optimizer(model)
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        loss_fn = TweetLoss(self.config)
        directory = Path(self.config.data.model_path)
        directory.mkdir(parents=True, exist_ok=True)
        log_file = directory / f'training_log_fold_{fold}.txt'
        log_file.write_text('')
        best_loss, patience = float('inf'), 0
        for epoch in range(self.config.model.max_epochs):
            model.train()
            total, count = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, desc=f'Fold {fold}, epoch {epoch + 1}', leave=False)):
                loss = self._batch_loss(model, batch, loss_fn)
                # Normalize the final partial accumulation window as well.
                window_start = (step // accumulation) * accumulation
                window_size = min(accumulation, len(train_loader) - window_start)
                scaler.scale(loss / window_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), self.config.model.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                size = len(batch['tokens'])
                total += loss.item() * size
                count += size
            val_loss = self._validate(model, valid_loader, loss_fn)
            if not math.isfinite(val_loss):
                raise ValueError('Validation loss is not finite')
            scheduler.step(val_loss)
            with log_file.open('a') as stream:
                stream.write(f'Epoch {epoch + 1} | Train Loss: {total / count:.4f} | Val Loss: {val_loss:.4f}\n')
            improved = val_loss < best_loss
            if improved or not self.config.training.save_best_only:
                self._save_checkpoint(model, fold, epoch, val_loss)
            if improved:
                best_loss, patience = val_loss, 0
            else:
                patience += 1
                if patience >= self.config.training.early_stopping_patience:
                    break
        return best_loss
