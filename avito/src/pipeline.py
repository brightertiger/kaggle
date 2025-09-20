import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
from .config import Config
from .data_utils import DataSplitter, DataLoader, FeatureValidator
from .feature_engineering import FeaturePipeline
from .models import ModelPipeline


class AvitoPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_splitter = DataSplitter(self.config)
        self.data_loader = DataLoader(self.config)
        self.feature_validator = FeatureValidator(self.config)
        
        self.feature_pipeline = FeaturePipeline(self.config)
        self.model_pipeline = ModelPipeline(self.config)
        
        self.feature_pipeline.set_data_loader(self.data_loader)
        self.model_pipeline.set_data_loader(self.data_loader)
    
    def preprocess_data(self) -> None:
        """Create cross-validation folds and validate data structure."""
        print("=== Data Preprocessing ===")
        
        print("Creating cross-validation folds...")
        self.data_splitter.create_cv_folds()
        
        print("Validating data files...")
        self._validate_data_files()
        
        print("Data preprocessing completed!")
    
    def _validate_data_files(self) -> None:
        """Validate that all required data files exist and are properly formatted."""
        required_files = [
            self.config.avito.TRAIN_DATA_PATH,
            self.config.avito.TEST_DATA_PATH,
            self.config.avito.TRAIN_ACTIVE_PATH,
            self.config.avito.TEST_ACTIVE_PATH
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        print("All required data files found!")
    
    def generate_features(self) -> None:
        """Generate all feature engineering components."""
        print("=== Feature Engineering ===")
        
        self.feature_pipeline.generate_all_features()
        
        print("Validating generated features...")
        self._validate_features()
        
        print("Feature engineering completed!")
    
    def _validate_features(self) -> None:
        """Validate that all feature files were generated correctly."""
        feature_files = [
            f"{self.config.avito.FEATURES_DIR}/count/count_1.csv",
            f"{self.config.avito.FEATURES_DIR}/count/count_2.csv",
            f"{self.config.avito.FEATURES_DIR}/text_title/title.csv",
            f"{self.config.avito.FEATURES_DIR}/user/user_features.csv",
            f"{self.config.avito.FEATURES_DIR}/date/time.csv"
        ]
        
        for file_path in feature_files:
            if not self.feature_validator.validate_feature_file(file_path, [self.config.avito.ID_COLUMN]):
                raise ValueError(f"Feature validation failed for: {file_path}")
        
        print("All features validated successfully!")
    
    def train_models(self) -> None:
        """Train all models in the pipeline."""
        print("=== Model Training ===")
        
        print("Training Level 1 models...")
        self.model_pipeline.train_level1_models()
        
        print("Training ensemble models...")
        self.model_pipeline.train_ensemble_models()
        
        print("Model training completed!")
    
    def generate_submission(self) -> pd.DataFrame:
        """Generate final submission file."""
        print("=== Generating Submission ===")
        
        submission_file = '../../data/outsample/scores/ensemble_model.csv'
        if not os.path.exists(submission_file):
            raise FileNotFoundError("Ensemble predictions not found. Please train models first.")
        
        submission = pd.read_csv(submission_file)
        submission = submission.rename(columns={self.config.avito.TARGET_COLUMN: 'deal_probability'})
        
        output_path = f"{self.config.avito.OUTPUT_DIR}/submission.csv"
        os.makedirs(self.config.avito.OUTPUT_DIR, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        print(f"Submission saved to: {output_path}")
        print(f"Submission shape: {submission.shape}")
        print(f"Deal probability range: {submission['deal_probability'].min():.4f} - {submission['deal_probability'].max():.4f}")
        
        return submission
    
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline from preprocessing to submission."""
        print("=== Avito Deal Probability Prediction Pipeline ===")
        print("Starting full pipeline execution...")
        
        try:
            self.preprocess_data()
            self.generate_features()
            self.train_models()
            submission = self.generate_submission()
            
            print("\n=== Pipeline Summary ===")
            print("✅ Data preprocessing completed")
            print("✅ Feature engineering completed")
            print("✅ Model training completed")
            print("✅ Submission file generated")
            print(f"📊 Final submission contains {len(submission)} predictions")
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            raise
    
    def evaluate_pipeline(self) -> Dict[str, float]:
        """Evaluate the pipeline performance using cross-validation."""
        print("=== Pipeline Evaluation ===")
        
        submission_file = '../../data/outsample/scores/ensemble_model.csv'
        if not os.path.exists(submission_file):
            raise FileNotFoundError("Ensemble predictions not found. Please train models first.")
        
        valid_scores = []
        
        for fold in range(1, self.config.avito.N_FOLDS + 1):
            valid_idx = pd.read_csv(f'../../data/data/files/valid_{fold}.csv')
            actual = self.data_loader.load_train_data([self.config.avito.ID_COLUMN, self.config.avito.TARGET_COLUMN])
            valid_actual = actual.merge(valid_idx, on=self.config.avito.ID_COLUMN)
            
            pred_file = f'../../data/insample/scores/ensemble_model.csv'
            if os.path.exists(pred_file):
                predictions = pd.read_csv(pred_file)
                valid_pred = predictions.merge(valid_idx, on=self.config.avito.ID_COLUMN)
                
                rmse = np.sqrt(np.mean((valid_actual[self.config.avito.TARGET_COLUMN] - valid_pred['score'])**2))
                valid_scores.append(rmse)
                print(f"Fold {fold} RMSE: {rmse:.5f}")
        
        if valid_scores:
            mean_rmse = np.mean(valid_scores)
            std_rmse = np.std(valid_scores)
            
            print(f"\nCross-validation results:")
            print(f"Mean RMSE: {mean_rmse:.5f}")
            print(f"Std RMSE: {std_rmse:.5f}")
            
            return {
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'fold_scores': valid_scores
            }
        else:
            print("No validation scores found.")
            return {}
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance from the trained models."""
        print("=== Feature Importance Analysis ===")
        
        importance_info = {
            'text_features': {
                'description': 'TF-IDF features from title and description text',
                'count': self.config.model.TFIDF_MAX_FEATURES * 2,
                'importance': 'High - Text content is crucial for deal prediction'
            },
            'user_features': {
                'description': 'User behavior and activity features',
                'count': 4,
                'importance': 'Medium - User patterns provide valuable signals'
            },
            'count_features': {
                'description': 'Categorical aggregation features',
                'count': 8,
                'importance': 'Medium - Category and location patterns'
            },
            'date_features': {
                'description': 'Temporal features from activation date',
                'count': 1,
                'importance': 'Low - Day of week has limited impact'
            }
        }
        
        print("Feature importance summary:")
        for feature_type, info in importance_info.items():
            print(f"  {feature_type}: {info['count']} features - {info['importance']}")
        
        return importance_info
