from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, log_loss
from ..utils.config import Config
from ..models.loss import WeightedBCELoss, reduce_loss


class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer,
                 scheduler, save_path, subset, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model = model.to(self.device)
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.loss_fn, self.optimizer, self.scheduler = loss_fn, optimizer, scheduler
        self.save_path, self.subset = Path(save_path), subset
        self.save_path.mkdir(parents=True, exist_ok=True)

    def save_model(self, epoch, loss, metric):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'backbone_config': self.model.xlmr.config.to_dict(),
            'tokenizer_config': {key: getattr(self.config, key) for key in
                                 ['TINY', 'MODEL_NAME', 'MAX_LENGTH', 'VOCAB_SIZE']},
            'loss': float(loss), 'metric': float(metric), 'epoch': epoch,
        }, self.save_path / f'model_{self.subset}.pt')

    def validate(self):
        self.model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch in self.valid_loader:
                labels.append(batch.pop('label').numpy().reshape(-1))
                batch.pop('weight', None)
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                scores.append(torch.sigmoid(self.model(**inputs)).cpu().numpy().reshape(-1))
        if not scores:
            raise ValueError('Validation dataset is empty')
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        loss = log_loss(labels, scores, labels=[0, 1])
        metric = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float('nan')
        self.model.train()
        return float(loss), float(metric)

    def train_epoch(self, epoch):
        self.model.train()
        self.train_loader.dataset.epoch(epoch)
        self.optimizer.zero_grad()
        losses = []
        steps = len(self.train_loader)
        accumulation = self.config.ACCUMULATION_STEPS
        for step, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}'), start=1):
            labels = batch.pop('label').to(self.device).float().reshape(-1, 1)
            weights = batch.pop('weight').to(self.device).float().reshape(-1, 1)
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            predictions = self.model(**inputs)
            raw_loss = (self.loss_fn(predictions, labels, weights)
                        if isinstance(self.loss_fn, WeightedBCELoss)
                        else self.loss_fn(predictions, labels))
            loss = reduce_loss(raw_loss)
            # Normalize each accumulation group, including a short final group.
            group_size = min(accumulation, steps - ((step - 1) // accumulation) * accumulation)
            (loss / group_size).backward()
            if step % accumulation == 0 or step == steps:
                clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                self.optimizer.zero_grad()
            losses.append(loss.detach().item())
        if not losses:
            raise ValueError('Training dataset is empty')
        return float(np.mean(losses))

    def train(self, epochs):
        if epochs < 1:
            raise ValueError('At least one epoch is required')
        best_metric = float('-inf')
        with (self.save_path / f'logfile_{self.subset}.txt').open('w', buffering=1) as logfile:
            for epoch in range(epochs):
                train_loss = self.train_epoch(epoch)
                valid_loss, valid_metric = self.validate()
                self.scheduler.step(valid_loss)
                selection_metric = valid_metric if np.isfinite(valid_metric) else -valid_loss
                if selection_metric > best_metric:
                    self.save_model(epoch, valid_loss, valid_metric)
                    best_metric = selection_metric
                logfile.write(f'Epoch {epoch} | Train Loss {train_loss:.4f} | '
                              f'Valid Loss {valid_loss:.4f} | Valid AUC {valid_metric:.4f}\n')
