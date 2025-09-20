import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from apex import amp

from src.config import Config
from src.data_utils import create_data_loaders, DiabeticRetinopathyDataset, NoiseAugmentedDataset
from src.model import DiabeticRetinopathyModel
from src.loss import DiabeticRetinopathyLoss, NoiseAugmentedLoss
from src.trainer import DiabeticRetinopathyTrainer, NoiseAugmentedTrainer
from src.optimizer import RAdam

class TrainingPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.trainer = DiabeticRetinopathyTrainer(self.config)
        self.noise_trainer = NoiseAugmentedTrainer(self.config)
    
    def pretrain_model(self, fold: int):
        print(f"Starting pretraining for fold {fold}")
        
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
    
    def train_model(self, fold: int):
        print(f"Starting training for fold {fold}")
        
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
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
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
    
    def combine_training(self, fold: int):
        print(f"Starting combined training for fold {fold}")
        
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

def main():
    config = Config()
    pipeline = TrainingPipeline(config)
    
    print("Starting APTOS Diabetic Retinopathy Training Pipeline")
    print("=" * 50)
    
    for fold in range(1, config.TRAIN_FOLDS + 1):
        print(f"\nTraining Fold {fold}")
        pipeline.combine_training(fold)

if __name__ == "__main__":
    main()
