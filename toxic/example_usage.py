#!/usr/bin/env python3
"""
Example usage of the Toxic Comment Classification Pipeline
"""

import os
import pandas as pd
from src.config import get_config
from src.pipeline import ToxicCommentPipeline
from src.data_utils import DataProcessor, TextPreprocessor
from src.models import NeuralNetworkModel, NaiveBayesSVM, LogisticRegressionModel
from src.ensemble import EnsembleModel, ModelEvaluator


def example_data_preprocessing():
    """Example of data preprocessing"""
    print("="*60)
    print("EXAMPLE: Data Preprocessing")
    print("="*60)
    
    config = get_config()
    data_processor = DataProcessor(config)
    
    # Example text samples
    sample_texts = [
        "This is a normal comment.",
        "You are an idiot!",
        "I hate this so much!!!",
        "This is actually helpful, thank you.",
        "Go die in a hole @user123"
    ]
    
    # Test different preprocessing methods
    text_series = pd.Series(sample_texts)
    
    for method in ['basic_clean', 'basic_clean_lower', 'tokenized', 'nltk_tokenized']:
        processed = TextPreprocessor.apply_preprocessing(text_series, method)
        print(f"\n{method.upper()}:")
        for original, processed_text in zip(sample_texts, processed):
            print(f"  Original: {original}")
            print(f"  Processed: {processed_text}")
            print()


def example_model_training():
    """Example of model training"""
    print("="*60)
    print("EXAMPLE: Model Training")
    print("="*60)
    
    config = get_config()
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'id': range(100),
        'comment_text': [
            f"This is sample comment number {i}. " + 
            ("It's toxic!" if i % 10 == 0 else "It's normal.")
            for i in range(100)
        ],
        'toxic': [1 if i % 10 == 0 else 0 for i in range(100)],
        'severe_toxic': [0] * 100,
        'obscene': [0] * 100,
        'threat': [0] * 100,
        'insult': [0] * 100,
        'identity_hate': [0] * 100
    })
    
    # Save sample data
    os.makedirs("data/raw", exist_ok=True)
    sample_data.to_csv("data/raw/train.csv", index=False)
    
    print("Created sample training data with 100 examples")
    print("Toxic comments: 10, Normal comments: 90")
    
    # Initialize pipeline
    pipeline = ToxicCommentPipeline(config)
    
    print("\nNote: This is a demonstration. Full training requires:")
    print("- Large dataset (100k+ examples)")
    print("- Pre-trained embeddings (GloVe/FastText)")
    print("- GPU for neural network training")
    print("- Several hours of compute time")


def example_ensemble_methods():
    """Example of ensemble methods"""
    print("="*60)
    print("EXAMPLE: Ensemble Methods")
    print("="*60)
    
    import numpy as np
    
    # Create sample predictions
    n_samples = 100
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Simulate predictions from different models
    model_predictions = {}
    
    for i in range(3):
        model_name = f"model_{i+1}"
        predictions = pd.DataFrame({
            'id': range(n_samples)
        })
        
        for col in target_columns:
            # Simulate predictions with some noise
            base_prob = 0.1 + i * 0.05  # Different base probabilities
            noise = np.random.normal(0, 0.02, n_samples)
            predictions[col] = np.clip(base_prob + noise, 0, 1)
        
        model_predictions[model_name] = predictions
    
    # Test ensemble methods
    config = get_config()
    ensemble_model = EnsembleModel(config)
    
    print("Sample predictions from 3 models:")
    for model_name, pred in model_predictions.items():
        print(f"{model_name} - toxic column mean: {pred['toxic'].mean():.3f}")
    
    # Simple averaging
    simple_ensemble = ensemble_model.simple_averaging(
        list(model_predictions.values()), target_columns
    )
    print(f"\nSimple ensemble - toxic column mean: {simple_ensemble['toxic'].mean():.3f}")
    
    # Weighted averaging
    weights = [0.5, 0.3, 0.2]  # Favor first model
    weighted_ensemble = ensemble_model.weighted_averaging(
        list(model_predictions.values()), weights, target_columns
    )
    print(f"Weighted ensemble - toxic column mean: {weighted_ensemble['toxic'].mean():.3f}")


def example_evaluation():
    """Example of model evaluation"""
    print("="*60)
    print("EXAMPLE: Model Evaluation")
    print("="*60)
    
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    
    # Create sample predictions and ground truth
    n_samples = 1000
    target_columns = ['toxic', 'severe_toxic', 'obscene']
    
    # Simulate ground truth (binary labels)
    ground_truth = pd.DataFrame({
        'id': range(n_samples),
        'toxic': np.random.binomial(1, 0.1, n_samples),
        'severe_toxic': np.random.binomial(1, 0.02, n_samples),
        'obscene': np.random.binomial(1, 0.05, n_samples)
    })
    
    # Simulate model predictions (probabilities)
    predictions = pd.DataFrame({
        'id': range(n_samples),
        'toxic': np.random.beta(2, 18, n_samples),  # Mean ~0.1
        'severe_toxic': np.random.beta(1, 49, n_samples),  # Mean ~0.02
        'obscene': np.random.beta(1, 19, n_samples)  # Mean ~0.05
    })
    
    # Evaluate
    config = get_config()
    evaluator = ModelEvaluator(config)
    
    results = evaluator.evaluate_single_model(
        predictions, ground_truth, target_columns
    )
    
    print("Model Performance:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")


def main():
    """Run all examples"""
    print("Toxic Comment Classification - Example Usage")
    print("="*60)
    
    try:
        example_data_preprocessing()
        example_model_training()
        example_ensemble_methods()
        example_evaluation()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed and paths are correct.")


if __name__ == "__main__":
    main()
