import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from sklearn.metrics import cohen_kappa_score
try:
    from apex import amp
except ImportError:
    amp = None

from .config import Config
from .loss import DiabeticRetinopathyLoss

class DiabeticRetinopathyTrainer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        requested = torch.device(self.config.DEVICE)
        self.device = torch.device('cpu') if requested.type == 'cuda' and not torch.cuda.is_available() else requested
        self.apex_enabled = False

    def initialize_amp(self, model, optimizer):
        self.apex_enabled = bool(self.config.USE_APEX and amp is not None and self.device.type == 'cuda')
        if self.apex_enabled:
            return amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True, verbosity=0)
        return model, optimizer

    def backward(self, loss, optimizer):
        if self.apex_enabled:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def calculate_quadratic_kappa(self, true_labels: np.ndarray, predictions: np.ndarray) -> float:
        true_labels = np.rint(true_labels)
        predictions = np.rint(predictions.clip(0., 4.))
        if len(np.unique(np.concatenate([true_labels, predictions]))) == 1:
            return float('nan')  # Kappa is undefined for a single constant category.
        return cohen_kappa_score(true_labels, predictions, labels=list(range(5)), weights='quadratic')

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.sum() / loss.shape[0]

    def save_model(self, epoch: int, model: torch.nn.Module, loss: float, path: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': float(loss),
            'config': asdict(self.config),
            'model_name': model.model_name,
            'image_size': model.inference_size
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def validate_model(self, model: torch.nn.Module, data_loader, loss_fn) -> tuple:
        model.eval()
        losses = []
        true_labels = []
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].float().to(self.device)
                labels = batch['label'].float().reshape(-1).to(self.device)
                weights = batch['weight'].float().reshape(-1).to(self.device)

                regression_output, classification_output = model(images)
                loss = self.reduce_loss(loss_fn(regression_output, classification_output, labels, weights))

                losses.extend([loss.item()] * labels.numel())
                true_labels.append(labels.detach().cpu().numpy())
                predictions.append(regression_output.detach().cpu().numpy())

        if not losses:
            raise ValueError('Data loader produced no batches')
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

        progress_bar = tqdm(total=len(data_loader.dataset),
                           ncols=0, disable=False)

        optimizer.zero_grad()
        for batch in data_loader:
            images = batch['image'].float().to(self.device)
            labels = batch['label'].float().reshape(-1).to(self.device)
            weights = batch['weight'].float().reshape(-1).to(self.device)

            regression_output, classification_output = model(images)
            loss = self.reduce_loss(loss_fn(regression_output, classification_output, labels, weights))

            self.backward(loss, optimizer)

            optimizer.step()
            optimizer.zero_grad()

            losses.extend([loss.item()] * labels.numel())
            true_labels.append(labels.detach().cpu().numpy())
            predictions.append(regression_output.detach().cpu().numpy())

            progress_bar.update(len(batch['idx']))
            progress_bar.set_postfix(trn_ls=f'{np.mean(losses):.5f}')

        if not losses:
            raise ValueError('Data loader produced no batches')
        true_labels = np.hstack(true_labels)
        predictions = np.hstack(predictions)
        kappa_score = self.calculate_quadratic_kappa(true_labels, predictions)

        progress_bar.close()
        return np.mean(losses), kappa_score

    def train_model(self, model: torch.nn.Module, train_loader, valid_loader,
                   loss_fn, optimizer, scheduler, save_path: str, epochs: int) -> None:
        if epochs < 1:
            raise ValueError('Training requires at least one epoch')
        model.to(self.device)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_kappa = self.train_epoch(model, train_loader, loss_fn, optimizer)
            valid_loss, valid_kappa = self.validate_model(model, valid_loader, loss_fn)

            if not np.isfinite(train_loss) or not np.isfinite(valid_loss):
                raise RuntimeError('Non-finite training or validation loss')
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
        # Validation has one deterministic view, no noisy labels or consistency term.
        validation_loss = DiabeticRetinopathyLoss(mse_weight=loss_fn.mse_weight, config=self.config)
        return super().validate_model(model, data_loader, validation_loss)

    def train_epoch(self, model: torch.nn.Module, data_loader, loss_fn, optimizer) -> tuple:
        model.train()
        losses = []
        true_labels = []
        predictions = []

        progress_bar = tqdm(total=len(data_loader.dataset),
                           ncols=0, disable=False)

        optimizer.zero_grad()
        for batch in data_loader:
            images_1 = batch['image_1'].float().to(self.device)
            images_2 = batch['image_2'].float().to(self.device)
            labels = batch['label'].float().reshape(-1).to(self.device)
            weights = batch['weight'].float().reshape(-1).to(self.device)

            reg_1, cls_1 = model(images_1)
            reg_2, cls_2 = model(images_2)

            loss = self.reduce_loss(loss_fn(reg_1, reg_2, cls_1, cls_2, labels, weights))

            self.backward(loss, optimizer)

            optimizer.step()
            optimizer.zero_grad()

            losses.extend([loss.item()] * labels.numel())
            true_labels.append(labels.detach().cpu().numpy())
            predictions.append(((reg_1 + reg_2) / 2).detach().cpu().numpy())

            progress_bar.update(len(batch['idx']))
            progress_bar.set_postfix(trn_ls=f'{np.mean(losses):.5f}')

        if not losses:
            raise ValueError('Data loader produced no batches')
        true_labels = np.hstack(true_labels)
        predictions = np.hstack(predictions)
        kappa_score = self.calculate_quadratic_kappa(true_labels, predictions)

        progress_bar.close()
        return np.mean(losses), kappa_score
