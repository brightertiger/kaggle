#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import IMetPipeline
from src.data_utils import DataPreprocessor, create_data_loaders
from src.models import ModelFactory, ResNextClassifier
from src.trainer import ModelTrainer
from src.scorer import ModelScorer, EnsembleScorer


def example_basic_usage():
    print("🎯 Example 1: Basic Pipeline Usage")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_name = 'resnext101'
    config.batch_size = 16
    config.epochs = 5
    
    pipeline = IMetPipeline(config)
    
    if pipeline.validate_setup():
        print("✅ Setup validation passed")
        
        print("🔄 Running preprocessing...")
        pipeline.preprocess_data()
        
        print("🚀 Training model (first 2 folds only)...")
        config.num_folds = 2
        results = pipeline.train_model()
        
        print("🔮 Generating predictions...")
        pipeline.generate_predictions()
        
        print("🎉 Basic pipeline completed!")
    else:
        print("❌ Setup validation failed")


def example_custom_training():
    print("\n🎯 Example 2: Custom Training Configuration")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_name = 'resnext50'
    config.batch_size = 32
    config.epochs = 10
    config.learning_rate = 5e-4
    config.image_size = 256
    config.patience = 3
    
    print(f"Configuration:\n{config}")
    
    trainer = ModelTrainer(config)
    
    print("🚀 Training single fold...")
    config.fold_idx = 1
    results = trainer.train_fold(1)
    
    print(f"✅ Training completed. Best F-Beta: {results['best_fbeta']:.4f}")


def example_data_exploration():
    print("\n🎯 Example 3: Data Exploration")
    print("=" * 50)
    
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    try:
        print("📊 Loading training data...")
        train_data = preprocessor.load_train_data()
        print(f"Training samples: {len(train_data)}")
        print(f"Columns: {list(train_data.columns)}")
        
        print("\n📊 Sample data:")
        print(train_data.head())
        
        print("\n📊 Label distribution:")
        attributes = ' '.join(train_data['attribute_ids'].tolist()).split()
        from collections import Counter
        attribute_counts = Counter(attributes)
        print(f"Total unique attributes: {len(attribute_counts)}")
        print(f"Most common attributes: {attribute_counts.most_common(10)}")
        
        print("\n📊 Creating folds...")
        folds_data = preprocessor.create_folds()
        print(f"Folds created: {folds_data['fold'].value_counts().sort_index().to_dict()}")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please ensure data files are in the correct location")


def example_model_inference():
    print("\n🎯 Example 4: Model Inference")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    
    scorer = ModelScorer(config)
    
    try:
        checkpoint_path = config.get_model_path(1, 'stage_2')
        
        if os.path.exists(checkpoint_path):
            print(f"🔮 Loading model from {checkpoint_path}")
            checkpoint_info = scorer.load_model(checkpoint_path)
            print(f"Model loaded: Loss={checkpoint_info['loss']:.4f}, "
                  f"Metric={checkpoint_info['metric']:.4f}")
            
            print("📊 Creating test loader...")
            from src.data_utils import create_test_loader
            test_loader = create_test_loader(config)
            
            print("🔮 Generating predictions...")
            predictions_df = scorer.generate_predictions(test_loader)
            
            print(f"✅ Generated predictions for {len(predictions_df)} samples")
            print(f"Prediction shape: {predictions_df.shape}")
            
        else:
            print(f"❌ Model checkpoint not found: {checkpoint_path}")
            print("Please train a model first")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")


def example_ensemble_prediction():
    print("\n🎯 Example 5: Ensemble Prediction")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    
    ensemble_scorer = EnsembleScorer(config)
    
    try:
        print("🔮 Scoring all folds...")
        score_files = ensemble_scorer.score_all_folds()
        
        if score_files:
            print(f"✅ Generated {len(score_files)} score files")
            
            print("📝 Creating submission...")
            submission_path = config.get_submission_path('example_submission')
            submission_df = ensemble_scorer.create_final_submission(
                score_files, submission_path
            )
            
            print(f"✅ Submission created with {len(submission_df)} predictions")
            print(f"Sample submission:\n{submission_df.head()}")
            
        else:
            print("❌ No score files generated. Please train models first")
            
    except Exception as e:
        print(f"❌ Ensemble prediction failed: {e}")


def example_weighted_ensemble():
    print("\n🎯 Example 6: Weighted Ensemble")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    
    pipeline = IMetPipeline(config)
    
    try:
        score_files = [
            config.get_score_path(1),
            config.get_score_path(2),
            config.get_score_path(3),
            config.get_score_path(4)
        ]
        
        weights = [0.3, 0.3, 0.2, 0.2]
        
        print(f"🔮 Creating weighted ensemble with weights: {weights}")
        
        submission_path = config.get_submission_path('weighted_ensemble')
        pipeline.generate_weighted_predictions(weights)
        
        print(f"✅ Weighted ensemble submission created")
        
    except Exception as e:
        print(f"❌ Weighted ensemble failed: {e}")


def example_model_comparison():
    print("\n🎯 Example 7: Model Comparison")
    print("=" * 50)
    
    models_to_compare = ['resnext50', 'resnext101']
    results = {}
    
    for model_name in models_to_compare:
        print(f"🚀 Training {model_name}...")
        
        config = Config()
        config.data_path = '../data'
        config.output_path = '../output'
        config.model_name = model_name
        config.batch_size = 16
        config.epochs = 3
        config.num_folds = 2
        
        trainer = ModelTrainer(config)
        fold_results = trainer.train_fold(1)
        
        results[model_name] = fold_results['best_fbeta']
        print(f"✅ {model_name} F-Beta Score: {fold_results['best_fbeta']:.4f}")
    
    print("\n📊 Model Comparison Results:")
    for model_name, score in results.items():
        print(f"  {model_name}: {score:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_model[0]} with F-Beta Score: {best_model[1]:.4f}")


def example_configuration_examples():
    print("\n🎯 Example 8: Configuration Examples")
    print("=" * 50)
    
    print("📋 Default Configuration:")
    default_config = Config()
    print(default_config)
    
    print("\n📋 Custom Configuration:")
    custom_config = Config()
    custom_config.model_name = 'resnext50'
    custom_config.batch_size = 32
    custom_config.epochs = 15
    custom_config.learning_rate = 1e-3
    custom_config.image_size = 224
    custom_config.focal_gamma = 2.0
    custom_config.f2_beta = 1.0
    
    print(custom_config)
    
    print("\n📋 Configuration as Dictionary:")
    config_dict = custom_config.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")


def main():
    print("🎨 iMet Collection 2019 - FGVC6 Competition Examples")
    print("=" * 60)
    
    examples = [
        example_basic_usage,
        example_custom_training,
        example_data_exploration,
        example_model_inference,
        example_ensemble_prediction,
        example_weighted_ensemble,
        example_model_comparison,
        example_configuration_examples
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
        
        if i < len(examples):
            print("\n" + "=" * 60)
    
    print("\n🎉 All examples completed!")
    print("\n💡 Tips:")
    print("  - Ensure data files are in the correct location")
    print("  - Adjust batch size based on available GPU memory")
    print("  - Use smaller number of folds for quick testing")
    print("  - Monitor GPU memory usage during training")


if __name__ == '__main__':
    main()
