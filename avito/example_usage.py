"""
Example usage of the Avito Deal Probability Prediction system.

This script demonstrates how to:
1. Load and preprocess data
2. Generate features
3. Train models
4. Make predictions
5. Evaluate performance
"""

import pandas as pd
import numpy as np
from src.config import Config
from src.pipeline import AvitoPipeline
from src.data_utils import DataLoader, FeatureValidator


def demonstrate_data_loading():
    """Demonstrate data loading and basic exploration."""
    print("=== Data Loading Demo ===")
    
    config = Config()
    data_loader = DataLoader(config)
    
    try:
        train_data = data_loader.load_train_data()
        test_data = data_loader.load_test_data()
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Training columns: {list(train_data.columns)}")
        print(f"Target column: {config.avito.TARGET_COLUMN}")
        print(f"Target range: {train_data[config.avito.TARGET_COLUMN].min():.3f} - {train_data[config.avito.TARGET_COLUMN].max():.3f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please ensure data files are in the correct location.")
        return False


def demonstrate_feature_engineering():
    """Demonstrate feature engineering pipeline."""
    print("\n=== Feature Engineering Demo ===")
    
    config = Config()
    pipeline = AvitoPipeline(config)
    
    try:
        print("Generating features...")
        pipeline.generate_features()
        print("✅ Feature engineering completed successfully!")
        
        validator = FeatureValidator(config)
        feature_files = [
            f"{config.avito.FEATURES_DIR}/count/count_1.csv",
            f"{config.avito.FEATURES_DIR}/text_title/title.csv",
            f"{config.avito.FEATURES_DIR}/user/user_features.csv"
        ]
        
        for file_path in feature_files:
            if validator.validate_feature_file(file_path):
                df = pd.read_csv(file_path)
                print(f"  {file_path}: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        return False


def demonstrate_model_training():
    """Demonstrate model training pipeline."""
    print("\n=== Model Training Demo ===")
    
    config = Config()
    pipeline = AvitoPipeline(config)
    
    try:
        print("Training models...")
        pipeline.train_models()
        print("✅ Model training completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Model training failed: {e}")
        return False


def demonstrate_evaluation():
    """Demonstrate model evaluation."""
    print("\n=== Model Evaluation Demo ===")
    
    config = Config()
    pipeline = AvitoPipeline(config)
    
    try:
        results = pipeline.evaluate_pipeline()
        
        if results:
            print(f"Cross-validation RMSE: {results['mean_rmse']:.5f} ± {results['std_rmse']:.5f}")
            print("✅ Model evaluation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return False


def demonstrate_submission_generation():
    """Demonstrate submission file generation."""
    print("\n=== Submission Generation Demo ===")
    
    config = Config()
    pipeline = AvitoPipeline(config)
    
    try:
        submission = pipeline.generate_submission()
        
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction range: {submission['deal_probability'].min():.4f} - {submission['deal_probability'].max():.4f}")
        print("✅ Submission generation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Submission generation failed: {e}")
        return False


def demonstrate_feature_importance():
    """Demonstrate feature importance analysis."""
    print("\n=== Feature Importance Demo ===")
    
    config = Config()
    pipeline = AvitoPipeline(config)
    
    try:
        importance = pipeline.get_feature_importance()
        
        print("Feature importance summary:")
        for feature_type, info in importance.items():
            print(f"  {feature_type}: {info['count']} features")
            print(f"    Description: {info['description']}")
            print(f"    Importance: {info['importance']}")
        
        return True
        
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    print("Avito Deal Probability Prediction - Example Usage")
    print("=" * 60)
    
    success_count = 0
    total_demos = 6
    
    if demonstrate_data_loading():
        success_count += 1
    
    if demonstrate_feature_engineering():
        success_count += 1
    
    if demonstrate_model_training():
        success_count += 1
    
    if demonstrate_evaluation():
        success_count += 1
    
    if demonstrate_submission_generation():
        success_count += 1
    
    if demonstrate_feature_importance():
        success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"✅ {success_count}/{total_demos} demonstrations completed successfully!")
    
    if success_count == total_demos:
        print("🎉 All demonstrations completed successfully!")
        print("📝 This example shows the complete usage of the refactored codebase.")
        print("🚀 For full pipeline execution, run: python main.py --step all")
    else:
        print("⚠️  Some demonstrations failed. Please check the error messages above.")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")
        print("💡 Ensure data files are in the correct location.")


if __name__ == "__main__":
    main()
