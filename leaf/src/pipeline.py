"""Coordinate data preparation, per-fold training, OOF scoring and blending."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from .data.data_utils import create_data_loaders, CassavaDataset
from .data.data_preprocessing import prepare_data, analyze_data, validate_image_ids
from .models.models import EfficientNetModel
from .models.loss import CELoss
from .training.trainer import train_model
from .training.inference import predict_with_tta
from .utils.config import Config
from .utils.ensemble import ModelEnsemble


class CassavaPipeline:
    def __init__(self, config=None):
        self.config = config or Config()
        self.data_dir = Path(self.config.DATA_DIR)
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.folds_path = self.output_dir / 'folds.csv'
        self.set_seed(self.config.SEED)

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_data(self, data_dir=None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        cfg = self.config
        data = prepare_data(self.data_dir, self.folds_path, cfg.N_FOLDS, cfg.SEED, cfg.NUM_CLASSES)
        analyze_data(data)
        return data

    def _folds(self):
        if not self.folds_path.exists():
            raise FileNotFoundError(f'{self.folds_path} missing; run prepare_data first')
        data = pd.read_csv(self.folds_path)
        if set(data['fold']) != set(range(self.config.N_FOLDS)):
            raise ValueError('Prepared folds differ from configured N_FOLDS; prepare data again')
        return data

    def _checkpoint_path(self, version, fold):
        if not version or version in {'.', '..'} or Path(version).name != version:
            raise ValueError('version must be a directory name')
        return self.output_dir / 'models' / version / f'model_{fold}.pt'

    def train_model(self, version, fold):
        cfg = self.config
        self._folds()
        if fold not in range(cfg.N_FOLDS):
            raise ValueError('fold is outside configured N_FOLDS')
        self.set_seed(cfg.SEED + fold)
        train_loader, valid_loader = create_data_loaders(
            self.data_dir / 'train_images', self.folds_path, fold, cfg)
        model = EfficientNetModel(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.PRETRAINED).to(cfg.DEVICE)
        optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        print(f'Training {version}, fold {fold} on {cfg.DEVICE}')
        history = train_model(
            model, train_loader, valid_loader, CELoss(), optimizer,
            self._checkpoint_path(version, fold), epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE, scheduler=scheduler, device=cfg.DEVICE,
            accumulation_steps=cfg.ACCUMULATION_STEPS, swa_start=cfg.SWA_START,
            patience=cfg.PATIENCE,
            metadata=dict(model_name=cfg.MODEL_NAME, num_classes=cfg.NUM_CLASSES,
                          image_size=cfg.IMAGE_SIZE, fold=fold, version=version))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return history

    def _load_model(self, version, fold):
        path = self._checkpoint_path(version, fold)
        if not path.is_file():
            raise FileNotFoundError(f'{path} missing; train every configured fold before scoring')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        swa_path = path.with_name(path.stem + '_swa.pt')
        if swa_path.exists():
            swa = torch.load(swa_path, map_location='cpu', weights_only=True)
            if swa['accuracy'] > checkpoint['accuracy']:
                checkpoint = swa
        metadata = checkpoint['metadata']
        if metadata['num_classes'] != self.config.NUM_CLASSES:
            raise ValueError('Checkpoint class count differs from configuration')
        # Loading trained weights must never trigger an ImageNet download.
        model = EfficientNetModel(metadata['model_name'], metadata['num_classes'], pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.config.DEVICE), metadata

    def _predict_frame(self, model, frame, image_dir, image_size):
        cfg = self.config
        dataset = CassavaDataset(image_dir, frame, is_train=False, image_size=image_size, seed=cfg.SEED)
        loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, shuffle=False)
        return predict_with_tta(model, loader, cfg.DEVICE, cfg.NUM_TTA)

    def _test_data(self, test_path=None):
        path = Path(test_path) if test_path else self.data_dir / 'sample_submission.csv'
        data = pd.read_csv(path)
        validate_image_ids(data)
        return data[['image_id']].copy()  # Sample-submission labels are placeholders.

    def score_model(self, version, test_path=None):
        cfg = self.config
        data, test_data = self._folds(), self._test_data(test_path)
        oof = np.full((len(data), cfg.NUM_CLASSES), np.nan)
        test_predictions = []
        for fold in range(cfg.N_FOLDS):
            model, metadata = self._load_model(version, fold)
            valid = data[data['fold'] == fold]
            oof[valid.index] = self._predict_frame(model, valid, self.data_dir / 'train_images', metadata['image_size'])
            test_predictions.append(self._predict_frame(model, test_data, self.data_dir / 'test_images', metadata['image_size']))
            del model
        columns = [f'prob_{i}' for i in range(cfg.NUM_CLASSES)]
        score_dir = self.output_dir / 'scores'
        score_dir.mkdir(parents=True, exist_ok=True)
        oof_frame = data[['image_id']].copy()
        oof_frame[columns] = oof
        oof_frame.to_csv(score_dir / f'{version}_oof.csv', index=False)
        test_probs = np.mean(test_predictions, axis=0)
        test_frame = test_data.copy()
        test_frame[columns] = test_probs
        test_frame.to_csv(score_dir / f'{version}_test.csv', index=False)
        submission = test_data.copy()
        submission['label'] = test_probs.argmax(axis=1)
        submission.to_csv(score_dir / f'{version}.csv', index=False)
        print(f'{version} OOF accuracy: {(oof.argmax(axis=1) == data["label"]).mean():.4f}')
        return submission

    def create_ensemble(self, versions=None, test_path=None):
        versions = versions or self.config.VERSIONS
        score_dir = self.output_dir / 'scores'
        ensemble = ModelEnsemble(self.output_dir / 'blend', self.config.NUM_CLASSES)
        results = ensemble.create_ensemble(
            self.folds_path,
            {version: score_dir / f'{version}_oof.csv' for version in versions},
            {version: score_dir / f'{version}_test.csv' for version in versions},
            self._test_data(test_path))
        print(f'Submission saved to {self.output_dir / "submission.csv"}')
        return results

    def run_full_pipeline(self, data_dir=None, test_path=None, versions=None):
        self.prepare_data(data_dir)
        versions = versions or self.config.VERSIONS
        for version in versions:
            for fold in range(self.config.N_FOLDS):
                self.train_model(version, fold)
            self.score_model(version, test_path)
        return self.create_ensemble(versions, test_path)
