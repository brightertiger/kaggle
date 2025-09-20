import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import os

from ..core import Config
from ..data import preprocess_all_data
from ..training import train_fold
from ..inference import generate_all_predictions

class IntracranialHemorrhagePipeline:
    """Main pipeline for intracranial hemorrhage detection"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def preprocess_data(self) -> None:
        """Run data preprocessing"""
        print("=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        preprocess_all_data(self.config)
        
        print("Data preprocessing completed successfully!")
    
    def train_models(self, model_name: str = 'resnext101') -> None:
        """Train models for all folds"""
        print("=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)
        
        if not self.config.TRAIN_CSV.exists():
            raise FileNotFoundError(f"Training data not found at {self.config.TRAIN_CSV}")
        
        training_histories = {}
        
        for fold_idx in range(1, self.config.NUM_FOLDS + 1):
            print(f"\nTraining fold {fold_idx}/{self.config.NUM_FOLDS}")
            
            history = train_fold(fold_idx, self.config, model_name)
            training_histories[f'fold_{fold_idx}'] = history
            
            print(f"Completed training fold {fold_idx}")
        
        print("\nAll folds trained successfully!")
        
        return training_histories
    
    def generate_predictions(self, model_name: str = 'resnext101') -> pd.DataFrame:
        """Generate predictions and create submission"""
        print("=" * 50)
        print("GENERATING PREDICTIONS")
        print("=" * 50)
        
        if not self.config.TEST_CSV.exists():
            raise FileNotFoundError(f"Test data not found at {self.config.TEST_CSV}")
        
        submission_df = generate_all_predictions(self.config, model_name)
        
        print("Predictions generated successfully!")
        
        return submission_df
    
    def run_full_pipeline(self, model_name: str = 'resnext101', 
                         skip_preprocessing: bool = False) -> pd.DataFrame:
        """Run complete pipeline"""
        
        if not skip_preprocessing:
            self.preprocess_data()
        
        self.train_models(model_name)
        
        submission_df = self.generate_predictions(model_name)
        
        print("=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        return submission_df

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Intracranial Hemorrhage Detection Pipeline')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'predict', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--model', type=str, default='resnext101', 
                       help='Model architecture')
    parser.add_argument('--skip-preprocess', action='store_true', 
                       help='Skip preprocessing step')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.device != 'auto':
        config.DEVICE = args.device
    
    pipeline = IntracranialHemorrhagePipeline(config)
    
    if args.mode == 'preprocess':
        pipeline.preprocess_data()
    elif args.mode == 'train':
        pipeline.train_models(args.model)
    elif args.mode == 'predict':
        pipeline.generate_predictions(args.model)
    elif args.mode == 'full':
        pipeline.run_full_pipeline(args.model, args.skip_preprocess)

if __name__ == "__main__":
    main()
