from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from apex import amp
except ImportError:
    amp = None
from .config import Config

class MelanomaTrainer:
    def __init__(self, model, train_loader, valid_loader, device=None, config=None):
        self.config = config or Config()
        device = device or self.config.DEVICE
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.8,
            patience=2,
            min_lr=1e-8
        )

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.POS_WEIGHT]).to(device)
        )

        self.use_apex = self.config.USE_APEX and amp is not None and torch.device(device).type == 'cuda'
        if self.use_apex:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O2', verbosity=False
            )
        if self.config.ACCUMULATION_STEPS < 1:
            raise ValueError('ACCUMULATION_STEPS must be positive')

        self.best_score = float('-inf')
        self.train_losses = []
        self.valid_losses = []
        self.valid_scores = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(self.train_loader, desc='Training')
        self.optimizer.zero_grad()
        accumulation = self.config.ACCUMULATION_STEPS

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            targets = batch['label'].to(self.device)

            outputs = self.model(images, metadata)
            loss = self.criterion(outputs, targets)

            window_start = (batch_idx // accumulation) * accumulation
            window_size = min(accumulation, len(self.train_loader) - window_start)
            backward_loss = loss / window_size
            if self.use_apex:
                with amp.scale_loss(backward_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                backward_loss.backward()

            if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(self.train_loader):
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
        if len(np.unique(melanoma_targets)) != 2:
            raise ValueError('Validation fold must contain both melanoma and non-melanoma labels')
        auc_score = float(roc_auc_score(melanoma_targets, melanoma_predictions))

        return avg_loss, auc_score, all_predictions, all_targets

    def train(self, epochs=Config.NUM_EPOCHS, save_path=None):
        if save_path is None:
            save_path = Config.MODEL_DIR / 'melanoma_model.pt'

        if epochs < 1:
            raise ValueError('epochs must be positive')
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        log_file = save_path.with_suffix('.csv')
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

        # Return the best validation checkpoint, not the last epoch's weights.
        self.load_model(save_path)
        print(f'\nTraining completed! Best AUC: {self.best_score:.4f}')
        return self.best_score

    def save_model(self, path, epoch, score):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': score,
            'config': self.config.as_dict(),
            'model_config': getattr(self.model, 'model_config', {}),
            'model_class': type(self.model).__name__
        }
        if self.use_apex:
            checkpoint['amp_state_dict'] = amp.state_dict()
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score = checkpoint['best_score']
        if self.use_apex and 'amp_state_dict' in checkpoint:
            amp.load_state_dict(checkpoint['amp_state_dict'])
        return checkpoint['epoch']
