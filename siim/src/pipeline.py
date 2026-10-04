import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .data_utils import create_data_loaders, diagnosis_class, load_metadata
from .inference import MelanomaInference, create_test_dataset
from .models import MelanomaClassifier
from .trainer import MelanomaTrainer


class MelanomaPipeline:
    def __init__(self, data_dir='data', model_dir='models', score_dir='scores', config=None):
        self.config = config or Config()
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.score_dir = Path(score_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.score_dir.mkdir(parents=True, exist_ok=True)
        self._set_seeds()
        self.train_metadata = None
        self.test_metadata = None
        self.models = []
        self.oof_predictions = None

    def _set_seeds(self):
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self):
        print('Loading metadata...')
        self.train_metadata, self.test_metadata = load_metadata(self.data_dir)
        n_folds = self.config.N_FOLDS
        if n_folds < 2:
            raise ValueError('N_FOLDS must be at least 2')
        if 'fold' not in self.train_metadata:
            labels = self.train_metadata.apply(diagnosis_class, axis=1)
            # Rare auxiliary diagnoses cannot be stratified; retain binary stratification.
            if labels.value_counts().min() < n_folds:
                labels = self.train_metadata['target']
            if labels.nunique() < 2 or labels.value_counts().min() < n_folds:
                raise ValueError('Each target class needs at least N_FOLDS images')
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.SEED)
            folds = np.full(len(self.train_metadata), -1, dtype=int)
            for fold, (_, val_idx) in enumerate(splitter.split(self.train_metadata, labels)):
                folds[val_idx] = fold
            self.train_metadata['fold'] = folds
        if set(self.train_metadata['fold']) != set(range(n_folds)):
            raise ValueError('fold must contain all integer fold IDs from 0 to N_FOLDS - 1')
        for fold in range(n_folds):
            validation = self.train_metadata.loc[self.train_metadata['fold'] == fold, 'target']
            if validation.nunique() != 2:
                raise ValueError(f'Fold {fold} must contain both target classes for ROC-AUC')
        self.oof_predictions = np.full(len(self.train_metadata), np.nan)
        print(f'Train metadata shape: {self.train_metadata.shape}')
        print(f'Test metadata shape: {self.test_metadata.shape}')

    def train_single_fold(self, fold, model_class=MelanomaClassifier, epochs=None):
        if self.train_metadata is None:
            self.load_data()
        if fold not in range(self.config.N_FOLDS):
            raise ValueError(f'Invalid fold: {fold}')
        epochs = self.config.NUM_EPOCHS if epochs is None else epochs
        print(f'\nTraining fold {fold}...')
        train_loader, valid_loader = create_data_loaders(
            self.data_dir / 'train', self.train_metadata, fold, config=self.config)
        model = model_class(model_name=self.config.MODEL_NAME,
                            num_classes=self.config.NUM_CLASSES,
                            metadata_dim=self.config.METADATA_DIM,
                            pretrained=self.config.PRETRAINED,
                            image_size=self.config.IMAGE_SIZE)
        trainer = MelanomaTrainer(model, train_loader, valid_loader, config=self.config)
        save_path = self.model_dir / f'melanoma_fold_{fold}.pt'
        best_score = trainer.train(epochs=epochs, save_path=save_path)
        inference = MelanomaInference(trainer.model, device=self.config.DEVICE)
        predictions = inference.predict(valid_loader, use_tta=False)[:, 1]
        self.oof_predictions[self.train_metadata['fold'] == fold] = predictions
        print(f'Fold {fold} completed. Best AUC: {best_score:.4f}')
        # Keep inactive folds off the GPU while training the next model.
        return trainer.model.to('cpu'), best_score

    def train_all_folds(self, model_class=MelanomaClassifier, epochs=None):
        if self.train_metadata is None:
            self.load_data()
        self.models = []
        fold_scores = []
        for fold in range(self.config.N_FOLDS):
            model, score = self.train_single_fold(fold, model_class, epochs)
            self.models.append(model)
            fold_scores.append(score)
        oof = self.train_metadata[['image_name', 'target', 'fold']].copy()
        oof['prediction'] = self.oof_predictions
        oof.to_csv(self.score_dir / 'oof.csv', index=False)
        print(f'Cross-validation AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')
        return fold_scores

    def predict_test_set(self, use_tta=True, ensemble_method='weighted_average'):
        if ensemble_method != 'weighted_average':
            raise ValueError('The pipeline supports fold averaging only. Fit EnsemblePredictor '
                             'separately on aligned held-out predictions for learned stacking.')
        if not self.models:
            raise ValueError('Train or load fold models before predicting')
        if self.test_metadata is None:
            self.load_data()
        test_dataset = create_test_dataset(self.data_dir / 'test', self.test_metadata, self.config)
        fold_predictions = []
        for fold, model in enumerate(self.models):
            print(f'Predicting with fold {fold} model...')
            inference = MelanomaInference(model, device=self.config.DEVICE)
            predictions = inference.predict_single_fold(
                test_dataset, batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.NUM_WORKERS, use_tta=use_tta)
            fold_predictions.append(predictions)
            model.to('cpu')
        # The original default is equal weights across fold models.
        final_predictions = np.mean(fold_predictions, axis=0)
        submission = pd.DataFrame({'image_name': self.test_metadata['image_name'],
                                   'target': final_predictions})
        submission_path = self.score_dir / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f'Predictions saved to {submission_path}')
        return final_predictions

    def run_full_pipeline(self, model_class=MelanomaClassifier, epochs=None, use_tta=True,
                          ensemble_method='weighted_average'):
        if ensemble_method != 'weighted_average':
            raise ValueError('Only weighted_average is supported by the full pipeline')
        self.load_data()
        fold_scores = self.train_all_folds(model_class, epochs)
        predictions = self.predict_test_set(use_tta, ensemble_method)
        return fold_scores, predictions
