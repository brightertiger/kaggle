import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import Config
from ..data.preprocessing import SaltDataProcessor
from ..data.data_utils import create_data_loaders
from ..training.trainer import ModelTrainer
from ..inference.predictor import ModelPredictor
from ..inference.evaluator import ModelEvaluator

class SaltSegmentationPipeline:
    """Main pipeline for Salt Identification from Aerial Images"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = SaltDataProcessor(config)
        self.predictor = ModelPredictor(config)
        self.evaluator = ModelEvaluator(config)
        
    def preprocess_data(self):
        """Run data preprocessing pipeline"""
        print("Starting data preprocessing...")
        self.data_processor.run_preprocessing()
        print("Data preprocessing completed!")
    
    def train_fold(self, fold_idx: int, model_name: str, resume: bool = False) -> Dict:
        """Train model for a specific fold"""
        print(f"Training fold {fold_idx} with model {model_name}")
        
        # Create data loaders
        train_loader, valid_loader = create_data_loaders(fold_idx, self.config)
        
        # Create trainer
        trainer = ModelTrainer(fold_idx, self.config)
        
        # Train model
        history = trainer.train(model_name, train_loader, valid_loader, resume)
        
        print(f"Training completed for fold {fold_idx}")
        return history
    
    def train_models(self, model_name: str, resume: bool = False) -> Dict[int, Dict]:
        """Train models for all folds"""
        print(f"Training {model_name} models for all folds...")
        
        all_histories = {}
        
        for fold_idx in range(1, self.config.NUM_FOLDS + 1):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold_idx}/{self.config.NUM_FOLDS}")
            print(f"{'='*50}")
            
            history = self.train_fold(fold_idx, model_name, resume)
            all_histories[fold_idx] = history
        
        # Print summary
        self._print_training_summary(all_histories)
        
        return all_histories
    
    def _print_training_summary(self, all_histories: Dict[int, Dict]):
        """Print training summary across all folds"""
        print(f"\n{'='*50}")
        print("TRAINING SUMMARY")
        print(f"{'='*50}")
        
        best_metrics = [history['best_metric'] for history in all_histories.values()]
        
        print(f"Average Best Metric: {np.mean(best_metrics):.4f}")
        print(f"Standard Deviation: {np.std(best_metrics):.4f}")
        print(f"Min Metric: {np.min(best_metrics):.4f}")
        print(f"Max Metric: {np.max(best_metrics):.4f}")
        
        print("\nIndividual Fold Results:")
        for fold_idx, history in all_histories.items():
            print(f"Fold {fold_idx}: {history['best_metric']:.4f}")
    
    def generate_predictions(self, model_name: str, use_tta: bool = False) -> Dict[int, np.ndarray]:
        """Generate predictions for all folds"""
        print(f"Generating predictions for {model_name}...")
        
        predictions = self.predictor.predict_all_folds(model_name, use_tta)
        
        print(f"Predictions generated for {len(predictions)} folds")
        return predictions
    
    def evaluate_models(self, model_name: str) -> Dict[str, float]:
        """Evaluate model performance across all folds"""
        print(f"Evaluating {model_name} model...")
        
        results = self.evaluator.evaluate_model_performance(model_name)
        
        return results
    
    def create_submission(self, model_name: str, use_tta: bool = False, 
                         threshold: Optional[float] = None) -> pd.DataFrame:
        """Create submission file"""
        print(f"Creating submission for {model_name}...")
        
        submission_df = self.predictor.create_submission(model_name, use_tta, threshold)
        
        print(f"Submission created with {len(submission_df)} predictions")
        return submission_df
    
    def run_full_pipeline(self, model_name: str, skip_preprocess: bool = False, 
                         resume: bool = False, use_tta: bool = False) -> pd.DataFrame:
        """Run complete pipeline from preprocessing to submission"""
        print("Running complete Salt Segmentation pipeline...")
        print(f"Model: {model_name}")
        print(f"Skip preprocessing: {skip_preprocess}")
        print(f"Resume training: {resume}")
        print(f"Use TTA: {use_tta}")
        print("=" * 60)
        
        # Step 1: Preprocess data (optional)
        if not skip_preprocess:
            self.preprocess_data()
        else:
            print("Skipping data preprocessing...")
        
        # Step 2: Train models
        print("\nStep 2: Training models...")
        self.train_models(model_name, resume)
        
        # Step 3: Generate predictions
        print("\nStep 3: Generating predictions...")
        self.generate_predictions(model_name, use_tta)
        
        # Step 4: Evaluate models
        print("\nStep 4: Evaluating models...")
        evaluation_results = self.evaluate_models(model_name)
        
        # Step 5: Create submission
        print("\nStep 5: Creating submission...")
        best_threshold = evaluation_results.get('best_threshold', self.config.IOU_CUTOFF)
        submission_df = self.create_submission(model_name, use_tta, best_threshold)
        
        print("\nPipeline completed successfully!")
        print(f"Final submission saved with {len(submission_df)} predictions")
        
        return submission_df
    
    def run_inference_only(self, model_name: str, use_tta: bool = False) -> pd.DataFrame:
        """Run inference only (assuming models are already trained)"""
        print("Running inference-only pipeline...")
        
        # Generate predictions
        self.generate_predictions(model_name, use_tta)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(model_name)
        
        # Create submission
        best_threshold = evaluation_results.get('best_threshold', self.config.IOU_CUTOFF)
        submission_df = self.create_submission(model_name, use_tta, best_threshold)
        
        return submission_df
