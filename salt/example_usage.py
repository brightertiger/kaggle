#!/usr/bin/env python3

"""
Example usage script for Salt Identification from Aerial Images project.
This script demonstrates various ways to use the pipeline for different tasks.
"""

import torch
import pandas as pd
from pathlib import Path
import numpy as np

from src.core import Config
from src.pipeline import SaltSegmentationPipeline
from src.data.preprocessing import SaltDataProcessor
from src.data.data_utils import create_data_loaders
from src.models.models import create_model
from src.training.trainer import ModelTrainer
from src.inference.predictor import ModelPredictor
from src.inference.evaluator import ModelEvaluator

def example_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("Example 1: Basic Usage")
    print("=" * 40)
    
    # Create configuration
    config = Config()
    
    # Create pipeline
    pipeline = SaltSegmentationPipeline(config)
    
    # Run full pipeline
    submission_df = pipeline.run_full_pipeline(model_name="seresnet34")
    
    print(f"Submission created with {len(submission_df)} predictions")
    print("First 5 predictions:")
    print(submission_df.head())

def example_custom_configuration():
    """Example 2: Custom configuration"""
    print("\nExample 2: Custom Configuration")
    print("=" * 40)
    
    # Create custom configuration
    config = Config()
    config.NUM_EPOCHS = 10  # Reduced for quick testing
    config.BATCH_SIZE_TRAIN = 16
    config.LEARNING_RATE = 0.0005
    config.IMAGE_SIZE = 101
    config.USE_FLIP_AUGMENTATION = True
    
    print("Custom configuration:")
    config.print_config()
    
    # Create pipeline with custom config
    pipeline = SaltSegmentationPipeline(config)
    
    # Run training only
    history = pipeline.train_models("seresnet34")
    
    print(f"Training completed for {len(history)} folds")

def example_single_fold_training():
    """Example 3: Train a single fold"""
    print("\nExample 3: Single Fold Training")
    print("=" * 40)
    
    config = Config()
    config.NUM_EPOCHS = 5  # Quick training for demo
    
    # Create data loaders for fold 1
    train_loader, valid_loader = create_data_loaders(1, config)
    
    # Create trainer
    trainer = ModelTrainer(1, config)
    
    # Train model
    history = trainer.train("seresnet34", train_loader, valid_loader)
    
    print(f"Training completed for fold 1")
    print(f"Best validation metric: {history['best_metric']:.4f}")

def example_data_preprocessing():
    """Example 4: Data preprocessing only"""
    print("\nExample 4: Data Preprocessing")
    print("=" * 40)
    
    config = Config()
    
    # Create data processor
    processor = SaltDataProcessor(config)
    
    # Run preprocessing
    processor.run_preprocessing()
    
    print("Data preprocessing completed!")

def example_model_inference():
    """Example 5: Model inference and evaluation"""
    print("\nExample 5: Model Inference")
    print("=" * 40)
    
    config = Config()
    
    # Create predictor and evaluator
    predictor = ModelPredictor(config)
    evaluator = ModelEvaluator(config)
    
    # Generate predictions for fold 1 (assuming model exists)
    try:
        valid_scores, test_scores = predictor.predict_single_fold(1, "seresnet34")
        print(f"Generated predictions: valid {valid_scores.shape}, test {test_scores.shape}")
        
        # Evaluate model
        metric = evaluator.evaluate_fold(1)
        print(f"Fold 1 metric: {metric:.4f}")
        
    except FileNotFoundError:
        print("Model not found. Please train models first.")

def example_custom_model():
    """Example 6: Create and use custom model"""
    print("\nExample 6: Custom Model")
    print("=" * 40)
    
    config = Config()
    
    # Create custom model
    model = create_model("seresnet34", config)
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

def example_batch_inference():
    """Example 7: Batch inference with TTA"""
    print("\nExample 7: Batch Inference with TTA")
    print("=" * 40)
    
    config = Config()
    predictor = ModelPredictor(config)
    
    try:
        # Generate predictions with Test Time Augmentation
        predictions = predictor.predict_all_folds("seresnet34", use_tta=True)
        
        # Create submission
        submission_df = predictor.create_submission("seresnet34", use_tta=True)
        
        print(f"Predictions generated for {len(predictions)} folds with TTA")
        print(f"Submission created with {len(submission_df)} predictions")
        
    except FileNotFoundError:
        print("Models not found. Please train models first.")

def example_threshold_optimization():
    """Example 8: Threshold optimization"""
    print("\nExample 8: Threshold Optimization")
    print("=" * 40)
    
    config = Config()
    evaluator = ModelEvaluator(config)
    
    try:
        # Find best threshold
        best_threshold, best_metric = evaluator.find_best_threshold()
        
        print(f"Best threshold: {best_threshold:.3f}")
        print(f"Best metric: {best_metric:.4f}")
        
    except ValueError:
        print("No predictions found. Please generate predictions first.")

def example_ensemble_prediction():
    """Example 9: Ensemble prediction from multiple folds"""
    print("\nExample 9: Ensemble Prediction")
    print("=" * 40)
    
    config = Config()
    
    # Load test indices
    test_path = config.PROCESSED_DATA_DIR / "test"
    if test_path.exists():
        import pickle
        with open(test_path / "test.pkl", 'rb') as f:
            test_indices = pickle.load(f)
        
        print(f"Test dataset has {len(test_indices)} images")
        
        # This would be used in the predictor for ensemble
        print("Ensemble prediction would combine predictions from all folds")
    else:
        print("Test data not found. Please run preprocessing first.")

def main():
    """Run all examples"""
    print("Salt Identification from Aerial Images - Usage Examples")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    # Run examples
    try:
        example_basic_usage()
        example_custom_configuration()
        example_single_fold_training()
        example_data_preprocessing()
        example_model_inference()
        example_custom_model()
        example_batch_inference()
        example_threshold_optimization()
        example_ensemble_prediction()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Some examples may require trained models or preprocessed data.")

if __name__ == "__main__":
    main()
