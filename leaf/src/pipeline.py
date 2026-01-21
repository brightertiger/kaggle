import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .data.data_utils import create_data_loaders, CassavaDataset
from torch.utils.data import DataLoader
from .data.data_preprocessing import prepare_data, analyze_data
from .models.models import EfficientNetModel
from .models.loss import CELoss
from .training.trainer import train_model
from .training.inference import predict
from .training.scoring import evaluate_model_performance
from .utils.config import Config
from .utils.ensemble import ModelEnsemble

class CassavaPipeline:
    def __init__(self):
        self.set_seed(Config.SEED)
        self.models = {}
        self.predictions = {}
    
    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    def prepare_data(self, data_dir):
        print("Preparing data...")
        data = prepare_data(data_dir)
        analyze_data(f"{data_dir}/raw/data.csv")
        return data
    
    def train_model(self, version, fold):
        print(f"Training {version} model for fold {fold}...")
        
        image_path = '../../data/merged/train_images/'
        label_path = '../../data/merged/data.csv'
        
        train_loader, valid_loader = create_data_loaders(image_path, label_path, fold)
        
        model = EfficientNetModel()
        model = model.to(Config.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        
        save_path = f'../../model/{version}/model_{fold}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=CELoss(),
            optimizer=optimizer,
            save_path=save_path,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            scheduler=scheduler
        )
        
        model.cpu()
        del model
        torch.cuda.empty_cache()
    
    def score_model(self, version, test_path):
        print(f"Scoring {version} model...")
        
        model_path = f'../../model/{version}/model_0.pt'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return
        
        model = EfficientNetModel()
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model = model.to(Config.DEVICE)
        
        test_data = pd.read_csv(test_path)
        test_dataset = CassavaDataset('../../data/test_images/', test_data, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        predictions = predict(model, test_loader)
        
        submission = pd.DataFrame({
            'image_id': test_data['image_id'],
            'label': np.argmax(predictions, axis=1)
        })
        
        output_path = f'../../score/{version}.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")
    
    def create_ensemble(self):
        print("Creating ensemble...")
        
        ensemble = ModelEnsemble()
        
        model_predictions = {
            'version0': '../../score/version0.csv',
            'version1': '../../score/version1.csv',
            'version2': '../../score/version2.csv',
            'version3': '../../score/version3.csv',
            'version4': '../../score/version4.csv',
            'version5': '../../score/version5.csv',
            'version6': '../../score/version6.csv',
            'version7': '../../score/version7.csv'
        }
        
        results = ensemble.create_ensemble('../../data/raw/data.csv', model_predictions)
        return results
    
    def run_full_pipeline(self, data_dir, test_path):
        print("Running full pipeline...")
        
        self.prepare_data(data_dir)
        
        for version in ['version0', 'version1', 'version2', 'version3', 'version4', 'version5', 'version6', 'version7']:
            for fold in range(5):
                self.train_model(version, fold)
            self.score_model(version, test_path)
        
        ensemble_results = self.create_ensemble()
        
        print("Pipeline completed successfully!")
        return ensemble_results
