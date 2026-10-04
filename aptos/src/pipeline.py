import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import random
import numpy as np
import pandas as pd

from .config import Config
from .data_utils import create_data_loaders, DiabeticRetinopathyDataset
from .model import DiabeticRetinopathyModel
from .loss import DiabeticRetinopathyLoss, NoiseAugmentedLoss
from .trainer import DiabeticRetinopathyTrainer, NoiseAugmentedTrainer
from .optimizer import RAdam

class APTOSPipeline:
    """Main pipeline for APTOS diabetic retinopathy detection."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        torch.manual_seed(self.config.RANDOM_SEED)
        self.trainer = DiabeticRetinopathyTrainer(self.config)
        self.noise_trainer = NoiseAugmentedTrainer(self.config)

        # Create output directories
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(self.config.MODEL_SAVE_PATH, "pretrain"), exist_ok=True)
        os.makedirs(os.path.join(self.config.MODEL_SAVE_PATH, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.config.MODEL_SAVE_PATH, "combine"), exist_ok=True)

    def preprocess_data(self):
        """Create cross-validation folds for all datasets."""
        print("Creating cross-validation folds...")

        from .preprocess import create_folds

        # External 2015 labels are optional for APTOS-only training.
        for labels, output in ((self.config.TRAIN_LABELS_2015, self.config.TRAIN_FOLDS_FILE),
                               (self.config.TEST_LABELS_2015, self.config.TEST_FOLDS_FILE)):
            source = Path(self.config.PRETRAIN_DATA_PATH) / labels
            if source.is_file():
                create_folds(source, source.with_name(output), self.config.PRETRAIN_FOLDS,
                             self.config.RANDOM_SEED)
            else:
                print(f'External labels absent: {source}; skipping their folds')
        source = Path(self.config.TRAIN_DATA_PATH) / self.config.TRAIN_LABELS_2019
        if not source.is_file() and source.name == 'train.csv':
            source = source.with_name('trainLabels19.csv')
        create_folds(source, source.with_name(self.config.TRAIN_FOLDS_FILE),
                     self.config.TRAIN_FOLDS, self.config.RANDOM_SEED)

    def _train_image_path(self):
        native = Path(self.config.TRAIN_DATA_PATH) / 'train_images'
        return str(native if native.is_dir() else native.with_name('train'))

    def pretrain_models(self):
        """Train models on 2015 pretrain data."""
        print("Starting pretraining phase...")

        for fold in range(1, self.config.PRETRAIN_FOLDS + 1):
            print(f"\nPretraining fold {fold}/{self.config.PRETRAIN_FOLDS}")
            self._pretrain_fold(fold)

    def train_models(self):
        """Train models on 2019 data with pretrained weights."""
        print("Starting training phase...")

        for fold in range(1, self.config.TRAIN_FOLDS + 1):
            print(f"\nTraining fold {fold}/{self.config.TRAIN_FOLDS}")
            self._train_fold(fold)

    def combine_training(self):
        """Train final models combining all datasets."""
        print("Starting combined training phase...")

        for fold in range(1, self.config.TRAIN_FOLDS + 1):
            print(f"\nCombined training fold {fold}/{self.config.TRAIN_FOLDS}")
            self._combine_fold(fold)

    def _pretrain_fold(self, fold: int):
        """Pretrain a single fold."""
        train_1, valid_1 = create_data_loaders(
            image_path=f"{self.config.PRETRAIN_DATA_PATH}/train",
            label_path=f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}",
            size=self.config.IMAGE_SIZE,
            fold_idx=fold,
            weight=1.0,
            config=self.config
        )

        train_2, valid_2 = create_data_loaders(
            image_path=f"{self.config.PRETRAIN_DATA_PATH}/test",
            label_path=f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TEST_FOLDS_FILE}",
            size=self.config.IMAGE_SIZE,
            fold_idx=fold,
            weight=1.0,
            config=self.config
        )

        train_dataset = ConcatDataset([train_1, train_2])
        valid_dataset = ConcatDataset([valid_1, valid_2])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            drop_last=False
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.VALIDATION_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            drop_last=False
        )

        model = DiabeticRetinopathyModel(self.config.MODEL_NAME, self.config)
        model = model.to(self.trainer.device)

        optimizer = RAdam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        model, optimizer = self.trainer.initialize_amp(model, optimizer)

        loss_fn = DiabeticRetinopathyLoss(
            mse_weight=self.config.MSE_WEIGHT,
            variance_weight=0.0,
            config=self.config
        )

        save_path = f"{self.config.MODEL_SAVE_PATH}/pretrain/model_{fold}.pt"

        self.trainer.train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=save_path,
            epochs=self.config.NUM_EPOCHS_PRETRAIN
        )

        model.cpu()
        del model
        torch.cuda.empty_cache()

    def _train_fold(self, fold: int):
        """Train a single fold."""
        train_dataset, valid_dataset = create_data_loaders(
            image_path=self._train_image_path(),
            label_path=f"{self.config.TRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}",
            size=self.config.IMAGE_SIZE,
            fold_idx=fold,
            weight=1.0,
            config=self.config
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            drop_last=False
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.VALIDATION_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            drop_last=False
        )

        pretrained_path = f"{self.config.MODEL_SAVE_PATH}/pretrain/model_1.pt"
        if not os.path.exists(pretrained_path):
            pretrained_path = f"{self.config.MODEL_SAVE_PATH}/pretrain/model_{fold}.pt"
        model = DiabeticRetinopathyModel(self.config.MODEL_NAME, self.config,
                                        pretrained=False if os.path.exists(pretrained_path) else None)
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained weights from {pretrained_path}")

        model = model.to(self.trainer.device)

        optimizer = RAdam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        model, optimizer = self.trainer.initialize_amp(model, optimizer)

        loss_fn = DiabeticRetinopathyLoss(
            mse_weight=self.config.MSE_WEIGHT,
            variance_weight=0.0,
            config=self.config
        )

        save_path = f"{self.config.MODEL_SAVE_PATH}/train/model_{fold}.pt"

        self.trainer.train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=save_path,
            epochs=self.config.NUM_EPOCHS_TRAIN
        )

        model.cpu()
        del model
        torch.cuda.empty_cache()

    def _combine_fold(self, fold: int):
        """Combine training for a single fold."""
        train_1, _ = create_data_loaders(
            image_path=f"{self.config.PRETRAIN_DATA_PATH}/train",
            label_path=f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}",
            size=self.config.LARGE_IMAGE_SIZE,
            fold_idx=fold,
            weight=1.0,
            use_noise_augmentation=self.config.USE_NOISE_AUGMENTATION,
            config=self.config
        )

        train_2, _ = create_data_loaders(
            image_path=f"{self.config.PRETRAIN_DATA_PATH}/test",
            label_path=f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TEST_FOLDS_FILE}",
            size=self.config.LARGE_IMAGE_SIZE,
            fold_idx=fold,
            weight=1.0,
            use_noise_augmentation=self.config.USE_NOISE_AUGMENTATION,
            config=self.config
        )

        train_3, valid_dataset = create_data_loaders(
            image_path=self._train_image_path(),
            label_path=f"{self.config.TRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}",
            size=self.config.LARGE_IMAGE_SIZE,
            fold_idx=fold,
            weight=5.0,
            use_noise_augmentation=self.config.USE_NOISE_AUGMENTATION,
            config=self.config
        )

        train_dataset = ConcatDataset([train_1, train_2, train_3])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            drop_last=False
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.VALIDATION_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            drop_last=False
        )

        model = DiabeticRetinopathyModel(self.config.MODEL_NAME, self.config)
        model.inference_size = self.config.LARGE_IMAGE_SIZE
        model = model.to(self.trainer.device)

        optimizer = RAdam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        trainer = self.noise_trainer if self.config.USE_NOISE_AUGMENTATION else self.trainer
        model, optimizer = trainer.initialize_amp(model, optimizer)

        loss_class = NoiseAugmentedLoss if self.config.USE_NOISE_AUGMENTATION else DiabeticRetinopathyLoss
        loss_fn = loss_class(
            mse_weight=self.config.MSE_WEIGHT,
            variance_weight=self.config.VARIANCE_WEIGHT,
            config=self.config
        )

        save_path = f"{self.config.MODEL_SAVE_PATH}/combine/model_{fold}.pt"

        trainer.train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=save_path,
            epochs=self.config.NUM_EPOCHS_COMBINE
        )

        model.cpu()
        del model
        torch.cuda.empty_cache()

    def run_full_pipeline(self, fold: int | None = None):
        """Run the complete training pipeline."""
        print("Starting APTOS Diabetic Retinopathy Training Pipeline")
        print("=" * 60)

        self.preprocess_data()
        if fold is None:
            self.pretrain_models()
            self.train_models()
            self.combine_training()
        else:
            self._pretrain_fold(fold)
            self._train_fold(fold)
            self._combine_fold(fold)

        print("\nPipeline completed successfully!")
        print("All models saved in the model directory.")

    def predict(self, checkpoints, test_csv=None, image_path=None, output_path=None):
        """Average continuous checkpoint predictions, then round to severity grades."""
        from dataclasses import replace

        if isinstance(checkpoints, (str, Path)):
            checkpoints = [checkpoints]
        if not checkpoints:
            raise ValueError('Prediction requires at least one checkpoint')
        test_csv = test_csv or Path(self.config.TRAIN_DATA_PATH) / 'test.csv'
        image_path = image_path or Path(self.config.TRAIN_DATA_PATH) / 'test_images'
        output_path = Path(output_path or Path(self.config.MODEL_SAVE_PATH) / 'submission.csv')
        data = pd.read_csv(test_csv, dtype={'id_code': str})
        if 'id_code' not in data or data.empty or data['id_code'].isna().any() or data['id_code'].duplicated().any():
            raise ValueError('test.csv needs nonempty, unique id_code values')
        predictions = []
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            settings = checkpoint.get('config', {})
            inference_config = replace(
                self.config,
                MODEL_NAME=checkpoint.get('model_name', self.config.MODEL_NAME),
                IMAGE_SIZE=checkpoint.get('image_size', self.config.IMAGE_SIZE),
                IMAGE_MEAN=tuple(settings.get('IMAGE_MEAN', self.config.IMAGE_MEAN)),
                IMAGE_STD=tuple(settings.get('IMAGE_STD', self.config.IMAGE_STD)),
                DROPOUT_RATE=settings.get('DROPOUT_RATE', self.config.DROPOUT_RATE),
                PRETRAINED=False,
            )
            model = DiabeticRetinopathyModel(inference_config.MODEL_NAME, inference_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.trainer.device).eval()
            dataset = DiabeticRetinopathyDataset(image_path, data[['id_code']],
                                                 inference_config.IMAGE_SIZE, config=inference_config)
            loader = DataLoader(dataset, batch_size=self.config.VALIDATION_BATCH_SIZE,
                                shuffle=False, num_workers=self.config.NUM_WORKERS)
            outputs = []
            with torch.no_grad():
                for batch in loader:
                    regression, _ = model(batch['image'].to(self.trainer.device))
                    outputs.append(regression.cpu().numpy())
            predictions.append(np.concatenate(outputs))
        continuous = np.mean(predictions, axis=0)
        if not np.isfinite(continuous).all():
            raise RuntimeError('Model produced non-finite predictions')
        submission = pd.DataFrame({'id_code': data['id_code'],
                                   'diagnosis': np.rint(continuous.clip(0, 4)).astype(int)})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f'Submission saved to {output_path} ({len(submission)} rows)')
        return submission
