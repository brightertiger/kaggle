import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data import DataLoader

from .config import Config
from .data_utils import MelanomaDataset, create_data_loaders, load_metadata
from .models import MelanomaClassifier, MelanomaClassifierV2
from .trainer import MelanomaTrainer
from .inference import MelanomaInference, create_test_dataset
from .ensemble import EnsemblePredictor

class MelanomaPipeline:
    def __init__(self, data_dir='data', model_dir='models', score_dir='scores'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.score_dir = Path(score_dir)
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.score_dir.mkdir(exist_ok=True)
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize components
        self.train_metadata = None
        self.test_metadata = None
        self.models = []
        self.predictions = []
    
    def _set_seeds(self):
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    def load_data(self):
        print("Loading metadata...")
        self.train_metadata, self.test_metadata = load_metadata(self.data_dir)
        
        # Add fold information
        skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
        folds = np.zeros(len(self.train_metadata))
        
        if 'diagnosis' in self.train_metadata.columns:
            for fold, (_, val_idx) in enumerate(skf.split(self.train_metadata, self.train_metadata['diagnosis'])):
                folds[val_idx] = fold
        else:
            # If no diagnosis column, use random folds
            folds = np.random.randint(0, Config.N_FOLDS, len(self.train_metadata))
        
        self.train_metadata['fold'] = folds
        
        print(f"Train metadata shape: {self.train_metadata.shape}")
        print(f"Test metadata shape: {self.test_metadata.shape}")
        print(f"Fold distribution: {np.bincount(folds.astype(int))}")
    
    def train_single_fold(self, fold, model_class=MelanomaClassifier, epochs=None):
        if epochs is None:
            epochs = Config.NUM_EPOCHS
        
        print(f"\nTraining fold {fold}...")
        
        # Create data loaders
        train_loader, valid_loader = create_data_loaders(
            self.data_dir / 'train',
            self.train_metadata,
            fold
        )
        
        # Initialize model
        model = model_class()
        
        # Initialize trainer
        trainer = MelanomaTrainer(model, train_loader, valid_loader)
        
        # Train model
        save_path = self.model_dir / f'melanoma_fold_{fold}.pt'
        best_score = trainer.train(epochs=epochs, save_path=save_path)
        
        print(f"Fold {fold} completed. Best AUC: {best_score:.4f}")
        
        return model, best_score
    
    def train_all_folds(self, model_class=MelanomaClassifier, epochs=None):
        print("Starting cross-validation training...")
        
        fold_scores = []
        
        for fold in range(Config.N_FOLDS):
            model, score = self.train_single_fold(fold, model_class, epochs)
            self.models.append(model)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\nCross-validation results:")
        print(f"Mean AUC: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Individual fold scores: {[f'{s:.4f}' for s in fold_scores]}")
        
        return fold_scores
    
    def predict_test_set(self, use_tta=True, ensemble_method='weighted_average'):
        print("Generating test predictions...")
        
        # Create test dataset
        test_dataset = create_test_dataset(
            self.data_dir / 'test',
            self.test_metadata
        )
        
        # Generate predictions for each fold
        fold_predictions = []
        
        for fold, model in enumerate(self.models):
            print(f"Predicting with fold {fold} model...")
            
            inference = MelanomaInference(model)
            predictions = inference.predict_single_fold(test_dataset, use_tta=use_tta)
            fold_predictions.append(predictions)
        
        # Convert to numpy array
        fold_predictions = np.array(fold_predictions).T
        
        # Create ensemble
        ensemble = EnsemblePredictor(method=ensemble_method)
        
        # For ensemble training, we need validation predictions
        # This is a simplified version - in practice, you'd use out-of-fold predictions
        print("Creating ensemble predictions...")
        
        if ensemble_method == 'weighted_average':
            # Simple weighted average based on fold performance
            weights = np.ones(Config.N_FOLDS) / Config.N_FOLDS
            final_predictions = np.average(fold_predictions, axis=1, weights=weights)
        else:
            # Use mean of all fold predictions
            final_predictions = np.mean(fold_predictions, axis=1)
        
        # Save predictions
        submission_df = pd.DataFrame({
            'image_name': self.test_metadata['image_name'],
            'target': final_predictions
        })
        
        submission_path = self.score_dir / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        print(f"Predictions saved to {submission_path}")
        print(f"Prediction statistics:")
        print(f"Mean: {final_predictions.mean():.4f}")
        print(f"Std: {final_predictions.std():.4f}")
        print(f"Min: {final_predictions.min():.4f}")
        print(f"Max: {final_predictions.max():.4f}")
        
        return final_predictions
    
    def run_full_pipeline(self, model_class=MelanomaClassifier, epochs=None, use_tta=True):
        print("Starting full melanoma classification pipeline...")
        
        # Load data
        self.load_data()
        
        # Train models
        fold_scores = self.train_all_folds(model_class, epochs)
        
        # Generate predictions
        predictions = self.predict_test_set(use_tta)
        
        print("\nPipeline completed successfully!")
        
        return fold_scores, predictions

def main():
    pipeline = MelanomaPipeline()
    fold_scores, predictions = pipeline.run_full_pipeline()
    
    print(f"\nFinal Results:")
    print(f"Cross-validation AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

if __name__ == "__main__":
    main()
