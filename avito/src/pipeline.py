import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from .config import Config
from .data_utils import DataSplitter, DataLoader, FeatureValidator, align_rows
from .feature_engineering import FeaturePipeline
from .models import ModelPipeline


class AvitoPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.config.avito.create_directories()
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
        self._validate_data_files()
        self.data_splitter.create_cv_folds()
        
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
        
        key, target = self.config.avito.ID_COLUMN, self.config.avito.TARGET_COLUMN
        train = self.data_loader.load_train_data([key, target])
        test = self.data_loader.load_test_data([key])
        for name, frame in [('train', train), ('test', test)]:
            if frame.empty or frame[key].isna().any() or frame[key].duplicated().any():
                raise ValueError(f'{name} must contain nonempty, unique item IDs')
        if set(train[key]) & set(test[key]):
            raise ValueError('Training and test item IDs overlap')
        if not train[target].between(0, 1).all():
            raise ValueError('Training targets must be finite probabilities in [0, 1]')
        if not 2 <= self.config.avito.N_FOLDS <= len(train):
            raise ValueError('N_FOLDS must be between 2 and the number of training rows')
        print("Required data files and training targets validated!")
    
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
        
        submission_file = f'{self.config.avito.OUTSAMPLE_DIR}/ensemble_model.csv'
        if not os.path.exists(submission_file):
            raise FileNotFoundError("Ensemble predictions not found. Please train models first.")
        
        submission = align_rows(
            self.data_loader.load_test_data([self.config.avito.ID_COLUMN]),
            pd.read_csv(submission_file), self.config.avito.ID_COLUMN)
        if not np.isfinite(submission[self.config.avito.TARGET_COLUMN]).all():
            raise ValueError('Submission contains non-finite predictions')
        submission[self.config.avito.TARGET_COLUMN] = submission[self.config.avito.TARGET_COLUMN].clip(0, 1)
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
    
    def evaluate_pipeline(self) -> Dict[str, Any]:
        """Report blend diagnostics; reused stacking folds are not nested CV."""
        key, target = self.config.avito.ID_COLUMN, self.config.avito.TARGET_COLUMN
        predictions = pd.read_csv(f'{self.config.avito.INSAMPLE_DIR}/ensemble_model.csv')
        predictions = predictions.rename(columns={target: 'prediction'})
        actual = self.data_loader.load_train_data([key, target])
        paired = align_rows(actual, predictions, key)
        scores = []
        for fold in range(1, self.config.avito.N_FOLDS + 1):
            _, valid_idx = self.data_loader.load_fold_data(fold)
            valid = align_rows(valid_idx, paired, key)
            rmse = float(np.sqrt(np.mean((valid[target] - valid['prediction']) ** 2)))
            scores.append(rmse)
            print(f'Fold {fold} blend diagnostic RMSE: {rmse:.5f}')
        return {'mean_rmse': float(np.mean(scores)), 'std_rmse': float(np.std(scores)),
                'fold_scores': scores}

    def get_feature_importance(self) -> Dict[str, Any]:
        """Compatibility alias for a feature inventory, not measured importance."""
        return {
            'text_features': {'count': 'vocabulary-dependent',
                              'description': 'Title/description TF-IDF and parameter counts',
                              'importance': 'Not measured; used by text Ridge'},
            'user_features': {'count': 4, 'description': 'Seller diversity and missingness',
                              'importance': 'Not measured; used by user Ridge'},
            'count_features': {'count': 8, 'description': 'Categorical frequencies',
                               'importance': 'Generated only; not used by the compact models'},
            'date_features': {'count': 1, 'description': 'Activation weekday',
                              'importance': 'Generated only; not used by the compact models'}
        }
