#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import MelanomaPipeline
from src.models import MelanomaClassifier, MelanomaClassifierV2
from src.data_utils import MelanomaDataset, load_metadata
from src.trainer import MelanomaTrainer
from src.inference import MelanomaInference
from src.ensemble import EnsemblePredictor
from src.config import Config

def example_basic_training():
    print("=== Basic Training Example ===")
    
    # Initialize pipeline
    pipeline = MelanomaPipeline(data_dir='data', model_dir='models', score_dir='scores')
    
    # Load data
    pipeline.load_data()
    
    # Train a single fold
    model, score = pipeline.train_single_fold(fold=0, epochs=5)
    print(f"Single fold training completed with AUC: {score:.4f}")

def example_cross_validation():
    print("\n=== Cross-Validation Example ===")
    
    # Initialize pipeline
    pipeline = MelanomaPipeline(data_dir='data', model_dir='models', score_dir='scores')
    
    # Run cross-validation
    fold_scores = pipeline.train_all_folds(epochs=5)
    print(f"Cross-validation completed with mean AUC: {np.mean(fold_scores):.4f}")

def example_custom_model():
    print("\n=== Custom Model Example ===")
    
    # Initialize pipeline with custom model
    pipeline = MelanomaPipeline(data_dir='data', model_dir='models', score_dir='scores')
    
    # Train with MelanomaClassifierV2
    fold_scores = pipeline.train_all_folds(model_class=MelanomaClassifierV2, epochs=5)
    print(f"Custom model training completed with mean AUC: {np.mean(fold_scores):.4f}")

def example_inference():
    print("\n=== Inference Example ===")
    
    # This would typically be done after training
    # For demonstration, we'll show the structure
    
    # Load metadata
    train_metadata, test_metadata = load_metadata(Path('data'))
    
    # Create test dataset
    test_dataset = MelanomaDataset(
        image_path=Path('data/test'),
        metadata_df=test_metadata,
        fold=None,
        is_training=False
    )
    
    print(f"Test dataset created with {len(test_dataset)} samples")
    print("Inference would be performed using MelanomaInference class")

def example_ensemble():
    print("\n=== Ensemble Example ===")
    
    # Create sample predictions (in practice, these would come from trained models)
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions from 3 different models
    model1_preds = np.random.beta(2, 5, n_samples)
    model2_preds = np.random.beta(3, 4, n_samples)
    model3_preds = np.random.beta(2.5, 4.5, n_samples)
    
    # Create ensemble
    ensemble = EnsemblePredictor(method='weighted_average')
    
    # Stack predictions
    stacked_preds = np.column_stack([model1_preds, model2_preds, model3_preds])
    
    # Simulate targets for demonstration
    targets = np.random.binomial(1, 0.1, n_samples)
    
    # Fit ensemble (in practice, you'd use validation data)
    ensemble.fit(stacked_preds, targets)
    
    # Make predictions
    final_preds = ensemble.predict(stacked_preds)
    
    print(f"Ensemble predictions generated for {len(final_preds)} samples")
    print(f"Mean prediction: {final_preds.mean():.4f}")

def example_full_pipeline():
    print("\n=== Full Pipeline Example ===")
    
    # Initialize pipeline
    pipeline = MelanomaPipeline(data_dir='data', model_dir='models', score_dir='scores')
    
    try:
        # Run complete pipeline
        fold_scores, predictions = pipeline.run_full_pipeline(
            model_class=MelanomaClassifier,
            epochs=5,  # Reduced for demo
            use_tta=True
        )
        
        print(f"Full pipeline completed successfully!")
        print(f"Cross-validation AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"Generated {len(predictions)} test predictions")
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please ensure data directory contains required files:")
        print("- data/train/ (training images)")
        print("- data/test/ (test images)")
        print("- data/train_metadata.csv")
        print("- data/test_metadata.csv")

def main():
    print("SIIM-ISIC Melanoma Classification - Example Usage")
    print("=" * 60)
    
    # Run examples
    try:
        example_basic_training()
        example_cross_validation()
        example_custom_model()
        example_inference()
        example_ensemble()
        example_full_pipeline()
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("This is expected if data files are not available.")
        print("Please ensure you have the required data structure.")

if __name__ == "__main__":
    main()
