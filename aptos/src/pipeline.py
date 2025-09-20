import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from apex import amp

from .config import Config
from .data_utils import create_data_loaders
from .model import DiabeticRetinopathyModel
from .loss import DiabeticRetinopathyLoss, NoiseAugmentedLoss
from .trainer import DiabeticRetinopathyTrainer, NoiseAugmentedTrainer
from .optimizer import RAdam

class APTOSPipeline:
    """Main pipeline for APTOS diabetic retinopathy detection."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
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
        
        # 2015 Pretrain data
        print("Processing 2015 Pretrain Data:")
        pretrain_train_path = f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TRAIN_LABELS_2015}"
        pretrain_train_output = f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}"
        create_folds(pretrain_train_path, pretrain_train_output, self.config.PRETRAIN_FOLDS)
        
        # 2015 Test data
        print("Processing 2015 Test Data:")
        pretrain_test_path = f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TEST_LABELS_2015}"
        pretrain_test_output = f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TEST_FOLDS_FILE}"
        create_folds(pretrain_test_path, pretrain_test_output, self.config.PRETRAIN_FOLDS)
        
        # 2019 Train data
        print("Processing 2019 Train Data:")
        train_path = f"{self.config.TRAIN_DATA_PATH}/{self.config.TRAIN_LABELS_2019}"
        train_output = f"{self.config.TRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}"
        create_folds(train_path, train_output, self.config.TRAIN_FOLDS)
        
        print("All folds created successfully!")
    
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
            drop_last=True
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.config.VALIDATION_BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS, 
            drop_last=True
        )
        
        model = DiabeticRetinopathyModel(self.config.MODEL_NAME, self.config)
        model = model.to(self.config.DEVICE)
        
        optimizer = RAdam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        
        model, optimizer = amp.initialize(
            model, optimizer, 
            opt_level="O2", 
            keep_batchnorm_fp32=True, 
            verbosity=0
        )
        
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
            image_path=f"{self.config.TRAIN_DATA_PATH}/train",
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
            drop_last=True
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS, 
            drop_last=True
        )
        
        model = DiabeticRetinopathyModel(self.config.MODEL_NAME, self.config)
        
        pretrained_path = f"{self.config.MODEL_SAVE_PATH}/pretrain/model_1.pt"
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained weights from {pretrained_path}")
        
        model = model.to(self.config.DEVICE)
        
        optimizer = RAdam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        
        model, optimizer = amp.initialize(
            model, optimizer, 
            opt_level="O2", 
            keep_batchnorm_fp32=True, 
            verbosity=0
        )
        
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
            config=self.config
        )
        
        train_2, _ = create_data_loaders(
            image_path=f"{self.config.PRETRAIN_DATA_PATH}/test",
            label_path=f"{self.config.PRETRAIN_DATA_PATH}/{self.config.TEST_FOLDS_FILE}",
            size=self.config.LARGE_IMAGE_SIZE,
            fold_idx=fold,
            weight=1.0,
            config=self.config
        )
        
        train_3, valid_dataset = create_data_loaders(
            image_path=f"{self.config.TRAIN_DATA_PATH}/train",
            label_path=f"{self.config.TRAIN_DATA_PATH}/{self.config.TRAIN_FOLDS_FILE}",
            size=self.config.LARGE_IMAGE_SIZE,
            fold_idx=fold,
            weight=5.0,
            config=self.config
        )
        
        train_dataset = ConcatDataset([train_1, train_2, train_3])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.config.NUM_WORKERS, 
            drop_last=True
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.config.VALIDATION_BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS, 
            drop_last=True
        )
        
        model = DiabeticRetinopathyModel(self.config.MODEL_NAME, self.config)
        model = model.to(self.config.DEVICE)
        
        optimizer = RAdam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        
        model, optimizer = amp.initialize(
            model, optimizer, 
            opt_level="O2", 
            keep_batchnorm_fp32=True, 
            verbosity=0
        )
        
        loss_fn = DiabeticRetinopathyLoss(
            mse_weight=self.config.MSE_WEIGHT, 
            variance_weight=self.config.VARIANCE_WEIGHT, 
            config=self.config
        )
        
        save_path = f"{self.config.MODEL_SAVE_PATH}/combine/model_{fold}.pt"
        
        self.trainer.train_model(
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
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("Starting APTOS Diabetic Retinopathy Training Pipeline")
        print("=" * 60)
        
        self.preprocess_data()
        self.pretrain_models()
        self.train_models()
        self.combine_training()
        
        print("\nPipeline completed successfully!")
        print("All models saved in the model directory.")
