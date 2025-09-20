#!/usr/bin/env python3

import torch
import pandas as pd
from pathlib import Path

from src.core import Config
from src.pipeline import IntracranialHemorrhagePipeline
from src.models import create_model
from src.data import create_data_loaders
from src.training import ModelTrainer
from src.inference import ModelPredictor, generate_all_predictions

def basic_usage_example():
    """Basic usage example of the intracranial hemorrhage detection pipeline"""
    
    print("Basic Usage Example")
    print("=" * 50)
    
    config = Config()
    
    pipeline = IntracranialHemorrhagePipeline(config)
    
    print("1. Data Preprocessing")
    pipeline.preprocess_data()
    
    print("\n2. Model Training")
    pipeline.train_models(model_name='resnext101')
    
    print("\n3. Generate Predictions")
    submission_df = pipeline.generate_predictions(model_name='resnext101')
    
    print(f"\nSubmission created with {len(submission_df)} predictions")
    print(submission_df.head())

def advanced_usage_example():
    """Advanced usage example with custom configurations"""
    
    print("Advanced Usage Example")
    print("=" * 50)
    
    config = Config()
    config.NUM_EPOCHS = 5
    config.BATCH_SIZE_TRAIN = 8
    config.LEARNING_RATE = 2e-4
    config.IMAGE_SIZE = 256
    
    pipeline = IntracranialHemorrhagePipeline(config)
    
    print("Custom configuration:")
    print(f"- Epochs: {config.NUM_EPOCHS}")
    print(f"- Batch Size: {config.BATCH_SIZE_TRAIN}")
    print(f"- Learning Rate: {config.LEARNING_RATE}")
    print(f"- Image Size: {config.IMAGE_SIZE}")
    
    submission_df = pipeline.run_full_pipeline(model_name='efficientnet', skip_preprocessing=True)
    
    return submission_df

def model_comparison_example():
    """Example comparing different model architectures"""
    
    print("Model Comparison Example")
    print("=" * 50)
    
    config = Config()
    config.NUM_EPOCHS = 2  # Reduced for quick comparison
    
    models_to_compare = ['resnet50', 'resnet101', 'inception']
    results = {}
    
    for model_name in models_to_compare:
        print(f"\nTraining {model_name}...")
        
        try:
            pipeline = IntracranialHemorrhagePipeline(config)
            pipeline.train_models(model_name)
            results[model_name] = "Training completed"
        except Exception as e:
            results[model_name] = f"Error: {e}"
    
    print("\nModel Comparison Results:")
    for model, result in results.items():
        print(f"- {model}: {result}")

def single_fold_training_example():
    """Example of training a single fold with custom parameters"""
    
    print("Single Fold Training Example")
    print("=" * 50)
    
    config = Config()
    config.NUM_EPOCHS = 3
    
    from src.training import train_fold
    
    fold_idx = 1
    model_name = 'resnext101'
    
    print(f"Training fold {fold_idx} with {model_name}")
    
    history = train_fold(fold_idx, config, model_name)
    
    print("Training completed!")
    print(f"Final validation loss: {history['valid_loss'][-1]:.5f}")
    print(f"Final validation metric: {history['valid_metric'][-1]:.5f}")

def inference_example():
    """Example of running inference on trained models"""
    
    print("Inference Example")
    print("=" * 50)
    
    config = Config()
    
    from src.inference import predict_fold, load_trained_model
    from src.data import create_inference_loader
    
    fold_idx = 1
    model_name = 'resnext101'
    
    model_path = config.MODEL_DIR / "best_model.pt"
    
    if model_path.exists():
        print(f"Loading trained model from {model_path}")
        
        model = load_trained_model(str(model_path), model_name, config)
        
        inference_loader = create_inference_loader(config)
        
        from src.inference import ModelPredictor
        predictor = ModelPredictor(model, config.DEVICE, config)
        
        results = predictor.predict_loader(inference_loader)
        
        print(f"Generated predictions for {len(results['predictions'])} samples")
        print(f"Prediction shape: {results['predictions'].shape}")
        
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first")

if __name__ == "__main__":
    print("RSNA Intracranial Hemorrhage Detection - Usage Examples")
    print("=" * 60)
    
    examples = {
        '1': ("Basic Usage", basic_usage_example),
        '2': ("Advanced Usage", advanced_usage_example),
        '3': ("Model Comparison", model_comparison_example),
        '4': ("Single Fold Training", single_fold_training_example),
        '5': ("Inference Only", inference_example),
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    choice = input("\nSelect an example to run (1-5): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        print("-" * 40)
        func()
    else:
        print("Invalid choice. Running basic usage example...")
        basic_usage_example()
