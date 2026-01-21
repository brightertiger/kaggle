"""
Example usage of the Statoil Iceberg Classifier Pipeline

This script demonstrates how to use the refactored pipeline for
training models and generating predictions.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import IcebergPipeline
from src.models import CNNBasic, CNNAdvanced, VGG16Model
from src.data_utils import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.config import Config

def example_full_pipeline():
    """Run the complete pipeline"""
    print("Running full pipeline...")
    pipeline = IcebergPipeline()
    pipeline.run_full_pipeline()

def example_individual_components():
    """Demonstrate individual pipeline components"""
    print("Running individual components...")
    
    # Initialize components
    config = Config()
    data_processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)
    
    # Prepare data
    print("Preparing data...")
    data_processor.process_train_data('source_1', data_processor.convert_images_source1)
    data_processor.process_test_data('source_1', data_processor.convert_images_source1)
    
    # Create XGBoost features
    print("Creating XGBoost features...")
    feature_engineer.create_xgboost_features(
        'data/download/train.json',
        'data/download/test.json'
    )

def example_model_training():
    """Demonstrate individual model training"""
    print("Training individual models...")
    
    from src.trainer import ModelTrainer
    
    config = Config()
    trainer = ModelTrainer(config)
    
    # Train CNN Basic model
    cnn_basic = CNNBasic(config)
    trainer.train_model(
        cnn_basic,
        'cnn_basic',
        'source_1',
        config.IMAGE_TRANSFORMS['source_1']
    )

def example_prediction():
    """Demonstrate prediction generation"""
    print("Generating predictions...")
    
    from src.predictor import ModelPredictor
    
    config = Config()
    predictor = ModelPredictor(config)
    
    # Generate predictions
    cnn_basic = CNNBasic(config)
    predictor.predict_test_set(cnn_basic, 'cnn_basic', 'source_1')
    predictor.predict_cv_set(cnn_basic, 'cnn_basic', 'source_1')

if __name__ == "__main__":
    print("Statoil Iceberg Classifier - Example Usage")
    print("=" * 50)
    
    # Choose which example to run
    example_choice = input("""
Choose an example to run:
1. Full Pipeline
2. Individual Components
3. Model Training
4. Prediction Generation
Enter choice (1-4): """)
    
    if example_choice == "1":
        example_full_pipeline()
    elif example_choice == "2":
        example_individual_components()
    elif example_choice == "3":
        example_model_training()
    elif example_choice == "4":
        example_prediction()
    else:
        print("Invalid choice. Running full pipeline...")
        example_full_pipeline()
