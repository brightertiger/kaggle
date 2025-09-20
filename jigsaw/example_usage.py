#!/usr/bin/env python3

"""
Example usage and demonstrations for Jigsaw Toxic Comment Classification.

This script demonstrates various ways to use the Jigsaw pipeline for
toxic comment classification with BERT and GPT models.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.pipeline import JigsawPipeline
from src.data_utils import DataProcessor, DataLoader
from src.models import BERTClassifier, GPTClassifier, ModelTrainer
from src.evaluation import ModelEvaluator


def example_basic_usage():
    """Example: Basic pipeline usage."""
    print("=" * 60)
    print("Example 1: Basic Pipeline Usage")
    print("=" * 60)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../model'
    
    pipeline = JigsawPipeline(config)
    
    if pipeline.validate_setup():
        print("✅ Setup validation passed")
        
        results = pipeline.run_full_pipeline(model_type='bert')
        print("✅ Pipeline completed successfully")
        
        return results
    else:
        print("❌ Setup validation failed")
        return None


def example_custom_configuration():
    """Example: Custom configuration and training parameters."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    config = Config()
    
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../model'
    
    config.random_seed = 123
    config.device = 'cuda:0'
    config.n_folds = 3
    config.max_length = 256
    
    config.bert_config.update({
        'learning_rate': 1e-5,
        'batch_size': 8,
        'num_epochs': 2,
        'dropout': 0.2
    })
    
    config.gpt_config.update({
        'learning_rate': 1e-5,
        'batch_size': 6,
        'num_epochs': 2,
        'dropout': 0.2
    })
    
    print("Configuration:")
    print(f"  Random Seed: {config.random_seed}")
    print(f"  Device: {config.device}")
    print(f"  Folds: {config.n_folds}")
    print(f"  Max Length: {config.max_length}")
    print(f"  BERT LR: {config.bert_config['learning_rate']}")
    print(f"  BERT Batch Size: {config.bert_config['batch_size']}")
    
    pipeline = JigsawPipeline(config)
    
    if pipeline.validate_setup():
        data_results = pipeline.process_data()
        print(f"✅ Processed {data_results['n_folds']} folds")
        
        model_results = pipeline.train_all_models('bert')
        print("✅ BERT models trained")
        
        return model_results
    else:
        print("❌ Setup validation failed")
        return None


def example_data_processing_only():
    """Example: Data processing without training."""
    print("=" * 60)
    print("Example 3: Data Processing Only")
    print("=" * 60)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    
    processor = DataProcessor(config)
    
    try:
        results = processor.process_all_data()
        
        print("Data Processing Results:")
        print(f"  Training samples: {len(results['train_data'])}")
        print(f"  Test samples: {len(results['test_data'])}")
        print(f"  CV folds: {results['n_folds']}")
        
        print("\nWeight distribution:")
        weight_dist = results['sample_weights']['weight'].value_counts().sort_index()
        for weight, count in weight_dist.items():
            print(f"  Weight {weight}: {count} samples")
        
        return results
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return None


def example_single_model_training():
    """Example: Training a single model on specific fold."""
    print("=" * 60)
    print("Example 4: Single Model Training")
    print("=" * 60)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../model'
    
    pipeline = JigsawPipeline(config)
    
    if not pipeline.validate_setup():
        print("❌ Setup validation failed")
        return None
    
    try:
        pipeline.process_data()
        print("✅ Data processed")
        
        fold_results = pipeline.train_bert_model(fold=1)
        print("✅ BERT model trained for fold 1")
        
        print(f"Best validation loss: {fold_results['best_loss']:.4f}")
        print(f"Predictions shape: {fold_results['predictions'].shape}")
        
        return fold_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def example_model_evaluation():
    """Example: Model evaluation and bias analysis."""
    print("=" * 60)
    print("Example 5: Model Evaluation")
    print("=" * 60)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../model'
    
    evaluator = ModelEvaluator(config)
    
    import pandas as pd
    import numpy as np
    
    predictions = pd.DataFrame({
        'id': range(100),
        'prediction': np.random.random(100)
    })
    
    ground_truth = pd.DataFrame({
        'id': range(100),
        'target': np.random.randint(0, 2, 100),
        'male': np.random.random(100),
        'female': np.random.random(100),
        'black': np.random.random(100),
        'white': np.random.random(100)
    })
    
    try:
        results = evaluator.evaluate_predictions(predictions, ground_truth)
        
        print("Evaluation Results:")
        print(f"  Overall AUC: {results['overall_auc']:.4f}")
        print(f"  Final Metric: {results['final_metric']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")
        print(f"  Optimal Threshold: {results['optimal_threshold']:.4f}")
        print(f"  Optimal F1: {results['optimal_f1']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None


def example_inference_only():
    """Example: Loading trained model and making predictions."""
    print("=" * 60)
    print("Example 6: Model Inference")
    print("=" * 60)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../model'
    
    pipeline = JigsawPipeline(config)
    
    try:
        test_predictions = pipeline.generate_test_predictions('bert')
        
        if not test_predictions.empty:
            print("Test Predictions:")
            print(f"  Total predictions: {len(test_predictions)}")
            print(f"  Prediction range: {test_predictions['prediction'].min():.4f} - {test_predictions['prediction'].max():.4f}")
            print(f"  Mean prediction: {test_predictions['prediction'].mean():.4f}")
            
            print("\nSample predictions:")
            print(test_predictions.head(10))
            
            return test_predictions
        else:
            print("❌ No predictions generated")
            return None
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None


def example_comparison_bert_vs_gpt():
    """Example: Comparing BERT vs GPT models."""
    print("=" * 60)
    print("Example 7: BERT vs GPT Comparison")
    print("=" * 60)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../model'
    
    config.bert_config.update({
        'num_epochs': 1,
        'batch_size': 8
    })
    
    config.gpt_config.update({
        'num_epochs': 1,
        'batch_size': 6
    })
    
    pipeline = JigsawPipeline(config)
    
    if not pipeline.validate_setup():
        print("❌ Setup validation failed")
        return None
    
    try:
        pipeline.process_data()
        print("✅ Data processed")
        
        print("\nTraining BERT model...")
        bert_results = pipeline.train_all_models('bert')
        
        print("\nTraining GPT model...")
        gpt_results = pipeline.train_all_models('gpt')
        
        print("\nEvaluating models...")
        evaluation_results = pipeline.evaluate_models()
        
        print("\nModel Comparison:")
        print("-" * 40)
        
        for fold_name, fold_results in evaluation_results.items():
            print(f"\n{fold_name}:")
            if 'bert' in fold_results:
                bert_metric = fold_results['bert']['final_metric']
                print(f"  BERT Final Metric: {bert_metric:.5f}")
            
            if 'gpt' in fold_results:
                gpt_metric = fold_results['gpt']['final_metric']
                print(f"  GPT Final Metric: {gpt_metric:.5f}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return None


def main():
    """Run all examples."""
    print("🚀 Jigsaw Toxic Comment Classification - Example Usage")
    print("=" * 80)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_configuration),
        ("Data Processing Only", example_data_processing_only),
        ("Single Model Training", example_single_model_training),
        ("Model Evaluation", example_model_evaluation),
        ("Inference Only", example_inference_only),
        ("BERT vs GPT Comparison", example_comparison_bert_vs_gpt)
    ]
    
    results = {}
    
    for name, example_func in examples:
        print(f"\n🔍 Running {name}...")
        try:
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    print("\n" + "=" * 80)
    print("📊 Example Summary")
    print("=" * 80)
    
    for name, result in results.items():
        status = "✅ Success" if result is not None else "❌ Failed"
        print(f"{name}: {status}")
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()
