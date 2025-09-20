import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.optimization import AdamW
from .utils.config import Config
from .models import XLMRobertaClassifier
from .data import create_data_loaders, create_test_loader
from .models import WeightedBCELoss
from .training import ModelTrainer
from .training import ModelInference
from .utils import AdversarialGenerator
from .utils import ModelEnsemble
from .data import EmbeddingProcessor
from .data import DataPreprocessor
from .training import ScoringPipeline

class JigsawPipeline:
    def __init__(self):
        self.config = Config()
        self.adversarial_generator = AdversarialGenerator()
        self.ensemble = ModelEnsemble(self.config.MODEL_DIR)
        self.embedding_processor = EmbeddingProcessor()
        self.data_preprocessor = DataPreprocessor()
        self.scoring_pipeline = ScoringPipeline(self.config.MODEL_DIR)
    
    def train_version1(self, subset, load_pretrained=False):
        torch.cuda.empty_cache()
        
        train_loader, valid_loader = create_data_loaders(subset)
        model = XLMRobertaClassifier()
        
        params = list(model.named_parameters())
        exc = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        param_groups = [
            {
                'params': [p for n, p in params if not any(ex in n for ex in exc)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in params if any(ex in n for ex in exc)],
                'weight_decay': 0.00
            }
        ]
        
        model = model.to(self.config.DEVICE)
        optimizer = AdamW(param_groups, lr=self.config.LR_V1)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=0)
        
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=WeightedBCELoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=f'{self.config.MODEL_DIR}/version1',
            subset=subset
        )
        
        trainer.train(self.config.EPOCHS_V1)
        
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
    
    def train_version2(self, subset, load_from_version1=True):
        torch.cuda.empty_cache()
        
        train_loader, valid_loader = create_data_loaders(subset)
        model = XLMRobertaClassifier()
        
        if load_from_version1:
            checkpoint_path = f'{self.config.MODEL_DIR}/version1/model_{subset}.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        params = list(model.named_parameters())
        exc = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        param_groups = [
            {
                'params': [p for n, p in params if not any(ex in n for ex in exc)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in params if any(ex in n for ex in exc)],
                'weight_decay': 0.00
            }
        ]
        
        model = model.to(self.config.DEVICE)
        optimizer = AdamW(param_groups, lr=self.config.LR_V2)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-7)
        
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss_fn=nn.BCEWithLogitsLoss(reduction='none'),
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=f'{self.config.MODEL_DIR}/version2',
            subset=subset
        )
        
        trainer.train(self.config.EPOCHS_V2)
        
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
    
    def prepare_data(self, data_dir):
        print("Preparing raw data...")
        self.data_preprocessor.process_all_data(data_dir)
    
    def generate_embeddings(self, data_dir):
        print("Generating embeddings...")
        self.embedding_processor.process_all_datasets(f"{data_dir}/process")
    
    def generate_adversarial_data(self, data_dir):
        print("Generating adversarial data...")
        self.adversarial_generator.generate_all_adversarial_data(f"{data_dir}/process")
    
    def create_ensemble(self):
        return self.ensemble.create_final_ensemble()
    
    def run_full_pipeline(self, data_dir, test_path):
        print("=== Starting Full Pipeline ===")
        
        print("Step 1: Preparing data...")
        self.prepare_data(data_dir)
        
        print("Step 2: Generating embeddings...")
        self.generate_embeddings(data_dir)
        
        print("Step 3: Generating adversarial data...")
        self.generate_adversarial_data(data_dir)
        
        print("Step 4: Training Version 1 models...")
        for subset in range(self.config.N_FOLDS):
            print(f"Training fold {subset}...")
            self.train_version1(subset)
        
        print("Step 5: Training Version 2 models...")
        for subset in range(self.config.N_FOLDS):
            print(f"Training fold {subset}...")
            self.train_version2(subset)
        
        print("Step 6: Scoring all models...")
        self.scoring_pipeline.score_all_models(test_path)
        
        print("Step 7: Creating ensemble...")
        final_predictions = self.create_ensemble()
        
        print("=== Pipeline Completed Successfully! ===")
        return final_predictions
