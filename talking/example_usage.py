#!/usr/bin/env python3

"""
Example usage script for TalkingData AdTracking Fraud Detection project.

This script demonstrates various ways to use the pipeline for different tasks
including data preprocessing, model training, prediction generation, and evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.core import Config
from src.pipeline import TalkingDataPipeline
from src.data.data_utils import TalkingDataProcessor, FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.models import TalkingDataModel, ModelEnsemble

def example_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("Example 1: Basic Usage")
    print("=" * 40)
    
    # Create configuration
    config = Config()
    
    # Create pipeline
    pipeline = TalkingDataPipeline(config)
    
    # Run full pipeline
    submission = pipeline.run_full_pipeline()
    
    print(f"Submission created with {len(submission)} predictions")
    print("First 5 predictions:")
    print(submission.head())

def example_custom_configuration():
    """Example 2: Custom configuration"""
    print("\nExample 2: Custom Configuration")
    print("=" * 40)
    
    # Create custom configuration
    config = Config()
    config.DATA_DIR = Path("../data")
    config.KEEP_HOURS = [4, 5, 9, 10, 13, 14]  # Keep only specific hours
    config.START_DATE = '2017-11-08 12:00:00'
    
    # Modify model parameters
    config.LGB_PARAMS['model_1']['learning_rate'] = 0.05
    config.LGB_PARAMS['model_1']['num_leaves'] = 64
    
    print(f"Custom configuration created:")
    print(f"  Data directory: {config.DATA_DIR}")
    print(f"  Keep hours: {config.KEEP_HOURS}")
    print(f"  Learning rate: {config.LGB_PARAMS['model_1']['learning_rate']}")

def example_data_preprocessing():
    """Example 3: Data preprocessing only"""
    print("\nExample 3: Data Preprocessing")
    print("=" * 35)
    
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # Run preprocessing
    preprocessor.preprocess_all_data()
    
    print("Data preprocessing completed!")
    print("Check the data/processed directory for output files")

def example_feature_engineering():
    """Example 4: Feature engineering demonstration"""
    print("\nExample 4: Feature Engineering")
    print("=" * 35)
    
    config = Config()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'ip': [1, 1, 2, 2, 3, 3],
        'app': [100, 100, 200, 200, 100, 200],
        'device': [1, 2, 1, 2, 1, 2],
        'os': [1, 1, 2, 2, 1, 2],
        'channel': [1, 2, 1, 2, 1, 2],
        'hour': [9, 10, 9, 10, 9, 10],
        'day': [9, 9, 9, 9, 9, 9]
    })
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Create count features
    count_features = feature_engineer.create_count_features(sample_data)
    
    print("Sample count features created:")
    for name, feature_df in list(count_features.items())[:3]:
        print(f"  {name}: {feature_df.shape}")
        print(f"    {feature_df.head()}")

def example_model_training():
    """Example 5: Model training demonstration"""
    print("\nExample 5: Model Training")
    print("=" * 30)
    
    config = Config()
    trainer = ModelTrainer(config)
    
    # Create sample training data
    n_samples = 1000
    train_data = pd.DataFrame({
        'click_id': range(n_samples),
        'ip': np.random.randint(1, 100, n_samples),
        'app': np.random.randint(1, 50, n_samples),
        'device': np.random.randint(1, 10, n_samples),
        'os': np.random.randint(1, 10, n_samples),
        'channel': np.random.randint(1, 20, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day': np.random.randint(8, 10, n_samples),
        'is_attributed': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    })
    
    valid_data = train_data.copy()
    valid_data['click_id'] += n_samples
    
    print(f"Sample training data: {train_data.shape}")
    print(f"Positive rate: {train_data['is_attributed'].mean():.4f}")
    
    # Train a single model
    metrics = trainer.train_single_model('model_1', train_data, valid_data)
    
    print(f"Model training completed:")
    print(f"  Train AUC: {metrics['train_auc']:.4f}")
    print(f"  Valid AUC: {metrics['valid_auc']:.4f}")

def example_prediction_generation():
    """Example 6: Prediction generation"""
    print("\nExample 6: Prediction Generation")
    print("=" * 35)
    
    config = Config()
    
    # Create sample test data
    test_data = pd.DataFrame({
        'click_id': range(100),
        'ip': np.random.randint(1, 100, 100),
        'app': np.random.randint(1, 50, 100),
        'device': np.random.randint(1, 10, 100),
        'os': np.random.randint(1, 10, 100),
        'channel': np.random.randint(1, 20, 100),
        'hour': np.random.randint(0, 24, 100),
        'day': np.random.randint(8, 10, 100)
    })
    
    # Create model (using dummy parameters for demonstration)
    model = TalkingDataModel(config, 'model_1')
    
    print(f"Sample test data: {test_data.shape}")
    print("Note: This is a demonstration - actual model would need to be trained first")

def example_ensemble_creation():
    """Example 7: Model ensemble creation"""
    print("\nExample 7: Model Ensemble")
    print("=" * 30)
    
    config = Config()
    ensemble = ModelEnsemble(config)
    
    # Create sample predictions
    n_samples = 100
    predictions = {}
    
    for i in range(1, 4):
        pred_data = pd.DataFrame({
            'click_id': range(n_samples),
            'is_attributed': np.random.random(n_samples)
        })
        predictions[f'score_{i}'] = pred_data
        print(f"Created sample predictions for model {i}")
    
    print("Sample ensemble weights:")
    for score, weight in config.ENSEMBLE_WEIGHTS.items():
        print(f"  {score}: {weight}")

def example_evaluation():
    """Example 8: Model evaluation"""
    print("\nExample 8: Model Evaluation")
    print("=" * 30)
    
    config = Config()
    
    # Create sample validation data with predictions
    valid_data = pd.DataFrame({
        'click_id': range(100),
        'is_attributed': np.random.choice([0, 1], 100, p=[0.99, 0.01]),
        'prediction': np.random.random(100)
    })
    
    print(f"Sample validation data: {valid_data.shape}")
    print(f"Positive rate: {valid_data['is_attributed'].mean():.4f}")
    
    # Calculate basic metrics
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(valid_data['is_attributed'], valid_data['prediction'])
    print(f"Sample AUC: {auc:.4f}")

def example_feature_importance():
    """Example 9: Feature importance analysis"""
    print("\nExample 9: Feature Importance")
    print("=" * 30)
    
    config = Config()
    
    # Create sample feature importance data
    features = ['ip_cnt', 'app_cnt', 'device_cnt', 'hour', 'day', 'channel_cnt']
    importance_values = np.random.random(len(features))
    importance_values = importance_values / importance_values.max()
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    print("Sample feature importance:")
    for i, (_, row) in enumerate(importance_df.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<15} {row['importance']:.4f}")

def example_data_analysis():
    """Example 10: Data analysis and visualization"""
    print("\nExample 10: Data Analysis")
    print("=" * 30)
    
    # Create sample data for analysis
    data = pd.DataFrame({
        'hour': np.random.randint(0, 24, 1000),
        'ip': np.random.randint(1, 100, 1000),
        'app': np.random.randint(1, 50, 1000),
        'is_attributed': np.random.choice([0, 1], 1000, p=[0.99, 0.01])
    })
    
    print("Sample data analysis:")
    print(f"  Total samples: {len(data)}")
    print(f"  Positive rate: {data['is_attributed'].mean():.4f}")
    print(f"  Unique IPs: {data['ip'].nunique()}")
    print(f"  Unique apps: {data['app'].nunique()}")
    
    # Hour distribution
    hour_dist = data['hour'].value_counts().sort_index()
    print(f"\nHour distribution (top 5):")
    for hour, count in hour_dist.head().items():
        print(f"  Hour {hour}: {count} clicks")

def main():
    """Run all examples"""
    print("TalkingData AdTracking Fraud Detection - Examples")
    print("=" * 55)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_configuration()
        example_data_preprocessing()
        example_feature_engineering()
        example_model_training()
        example_prediction_generation()
        example_ensemble_creation()
        example_evaluation()
        example_feature_importance()
        example_data_analysis()
        
        print("\n" + "=" * 55)
        print("All examples completed successfully!")
        print("\nTo run the actual pipeline:")
        print("  python main.py --mode full --data-dir ../data")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
