import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .core.config import Config
from .data.preprocessing import DataPreprocessor
from .data.data_utils import TalkingDataProcessor
from .models.trainer import ModelTrainer
from .models.models import ModelEnsemble

class TalkingDataPipeline:
    """Main pipeline for TalkingData AdTracking fraud detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.trainer = ModelTrainer(config)
        self.ensemble = ModelEnsemble(config)
        
    def run_full_pipeline(self, skip_preprocess: bool = False) -> pd.DataFrame:
        """Run the complete pipeline from preprocessing to submission."""
        
        print("Starting TalkingData AdTracking Fraud Detection Pipeline")
        print("=" * 60)
        
        # Step 1: Data preprocessing
        if not skip_preprocess:
            print("\n1. Data Preprocessing")
            print("-" * 30)
            self.preprocess_data()
        
        # Step 2: Feature engineering and merging
        print("\n2. Feature Engineering and Data Merging")
        print("-" * 40)
        self.create_feature_engineered_data()
        
        # Step 3: Model training
        print("\n3. Model Training")
        print("-" * 20)
        self.train_models()
        
        # Step 4: Generate predictions
        print("\n4. Generating Predictions")
        print("-" * 30)
        self.generate_predictions()
        
        # Step 5: Create ensemble
        print("\n5. Creating Ensemble")
        print("-" * 25)
        submission = self.create_ensemble_submission()
        
        print("\nPipeline completed successfully!")
        print(f"Final submission has {len(submission)} predictions")
        
        return submission
    
    def preprocess_data(self):
        """Run data preprocessing pipeline."""
        self.preprocessor.preprocess_all_data()
    
    def create_feature_engineered_data(self):
        """Create feature-engineered datasets for training and testing."""
        
        # Load processed data
        train_data = pd.read_feather(self.config.PROCESSED_DATA_DIR / 'train_data.feather')
        valid_data = pd.read_feather(self.config.PROCESSED_DATA_DIR / 'valid_data.feather')
        test_data = pd.read_feather(self.config.PROCESSED_DATA_DIR / 'test_data.feather')
        
        # Merge with features
        train_data = self._merge_features(train_data, 'train')
        valid_data = self._merge_features(valid_data, 'valid')
        test_data = self._merge_features(test_data, 'test')
        
        # Save feature-engineered data
        train_data.to_feather(self.config.MODELS_DIR / 'train_data.feather')
        valid_data.to_feather(self.config.MODELS_DIR / 'valid_data.feather')
        test_data.to_feather(self.config.MODELS_DIR / 'test_data.feather')
        
        print(f"Feature-engineered data saved:")
        print(f"  Train: {train_data.shape}")
        print(f"  Valid: {valid_data.shape}")
        print(f"  Test: {test_data.shape}")
    
    def train_models(self):
        """Train multiple LightGBM models."""
        
        # Load data
        train_data = pd.read_feather(self.config.MODELS_DIR / 'train_data.feather')
        valid_data = pd.read_feather(self.config.MODELS_DIR / 'valid_data.feather')
        
        print(f"Training data: {train_data.shape}")
        print(f"Validation data: {valid_data.shape}")
        print(f"Positive rate (train): {train_data[self.config.TARGET_COLUMN].mean():.4f}")
        print(f"Positive rate (valid): {valid_data[self.config.TARGET_COLUMN].mean():.4f}")
        
        # Train models
        model_names = ['model_1', 'model_2']
        metrics = self.trainer.train_multiple_models(train_data, valid_data, model_names)
        
        # Print summary
        print("\nModel Training Summary:")
        for metric in metrics:
            print(f"  {metric['model_name']}: Train AUC = {metric['train_auc']:.4f}, "
                  f"Valid AUC = {metric['valid_auc']:.4f}")
    
    def generate_predictions(self):
        """Generate predictions for validation and test data."""
        
        # Load data
        valid_data = pd.read_feather(self.config.MODELS_DIR / 'valid_data.feather')
        test_data = pd.read_feather(self.config.MODELS_DIR / 'test_data.feather')
        
        # Create output directories
        score_dir = self.config.DATA_DIR / 'score'
        submit_dir = self.config.DATA_DIR / 'submit'
        score_dir.mkdir(exist_ok=True)
        submit_dir.mkdir(exist_ok=True)
        
        # Generate predictions for each model
        model_names = ['model_1', 'model_2']
        
        for model_name in model_names:
            # Validation predictions
            valid_pred = self.trainer.generate_predictions(
                valid_data, model_name, 
                score_dir / f'{model_name}.csv'
            )
            
            # Test predictions
            test_pred = self.trainer.create_submission(
                test_data, model_name,
                submit_dir / f'{model_name}.csv'
            )
            
            print(f"Predictions generated for {model_name}")
    
    def create_ensemble_submission(self) -> pd.DataFrame:
        """Create final ensemble submission."""
        
        # Create submission directory
        submit_dir = self.config.DATA_DIR / 'submit'
        
        # List of model prediction files
        model_files = [submit_dir / f'model_{i}.csv' for i in range(1, 3)]
        
        # Create ensemble
        ensemble_submission = self.ensemble.blend_predictions(
            model_files, 
            submit_dir / 'ensemble.csv',
            mode='submit'
        )
        
        # Save final submission
        final_submission_path = self.config.SUBMISSIONS_DIR / 'final_submission.csv'
        ensemble_submission.to_csv(final_submission_path, index=False)
        
        print(f"Final ensemble submission saved to {final_submission_path}")
        
        return ensemble_submission
    
    def _merge_features(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Merge data with engineered features."""
        
        print(f"Merging features for {mode} data...")
        
        # Load count features
        count_features = self._load_count_features()
        
        # Load unique features
        unique_features = self._load_unique_features()
        
        # Load ranking features
        ranking_features = self._load_ranking_features()
        
        # Merge count features
        for name, feature_df in count_features.items():
            if name == 'user_count':
                data = data.merge(feature_df, on=['ip', 'device', 'os'], how='left')
            elif name == 'user_app_count':
                data = data.merge(feature_df, on=['ip', 'device', 'os', 'app'], how='left')
            else:
                # Extract feature name and columns for merging
                feature_cols = [col for col in feature_df.columns if col != name]
                data = data.merge(feature_df, on=feature_cols, how='left')
        
        # Merge unique features
        for name, feature_df in unique_features.items():
            if 'ip_' in name:
                data = data.merge(feature_df, on='ip', how='left')
        
        # Merge ranking features
        for name, feature_df in ranking_features.items():
            if name == 'ip_rank':
                data = data.merge(feature_df, on='ip', how='left')
            elif 'app_channel' in name:
                data = data.merge(feature_df, on=['app', 'channel'], how='left')
            elif 'app_os' in name:
                data = data.merge(feature_df, on=['app', 'os'], how='left')
            elif 'channel_os' in name:
                data = data.merge(feature_df, on=['channel', 'os'], how='left')
        
        # Fill missing values for ranking features
        ranking_cols = [col for col in data.columns if 'rank' in col]
        data[ranking_cols] = data[ranking_cols].fillna(11)
        
        # Fill other missing values
        data = data.fillna(0)
        
        # Optimize data types
        data = self._optimize_dtypes(data)
        
        return data
    
    def _load_count_features(self) -> Dict[str, pd.DataFrame]:
        """Load count-based features."""
        features = {}
        
        count_files = [
            'ip', 'app', 'os', 'ip_day_hour', 'ip_app', 'ip_app_os',
            'ip_device', 'app_channel', 'ip_hour_os', 'ip_hour_app',
            'user_count', 'user_app_count'
        ]
        
        for filename in count_files:
            file_path = self.config.FEATURES_DIR / 'count' / f'{filename}.feather'
            if file_path.exists():
                features[f'{filename}_cnt'] = pd.read_feather(file_path)
        
        return features
    
    def _load_unique_features(self) -> Dict[str, pd.DataFrame]:
        """Load unique count features."""
        features = {}
        
        unique_files = ['ip_app_unq', 'ip_channel_unq']
        
        for filename in unique_files:
            file_path = self.config.FEATURES_DIR / 'unique' / f'{filename}.feather'
            if file_path.exists():
                features[filename] = pd.read_feather(file_path)
        
        return features
    
    def _load_ranking_features(self) -> Dict[str, pd.DataFrame]:
        """Load ranking features."""
        features = {}
        
        rank_files = ['ip', 'app_channel', 'app_os', 'channel_os']
        
        for filename in rank_files:
            file_path = self.config.FEATURES_DIR / 'rank' / f'{filename}.feather'
            if file_path.exists():
                features[f'{filename}_rank'] = pd.read_feather(file_path)
        
        return features
    
    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        for column in data.columns:
            if data[column].dtype == 'int64':
                if data[column].max() <= 250:
                    data[column] = data[column].astype('uint8')
                elif data[column].max() <= 65000 and data[column].min() >= 0:
                    data[column] = data[column].astype('uint16')
                elif data[column].max() <= 4294967295 and data[column].min() >= 0:
                    data[column] = data[column].astype('uint32')
        
        return data
    
    def evaluate_models(self) -> Dict:
        """Evaluate all trained models."""
        
        # Load validation data
        valid_data = pd.read_feather(self.config.MODELS_DIR / 'valid_data.feather')
        
        # Evaluate each model
        model_names = ['model_1', 'model_2']
        evaluations = {}
        
        for model_name in model_names:
            evaluation = self.trainer.evaluate_model(valid_data, model_name)
            evaluations[model_name] = evaluation
        
        return evaluations
    
    def get_feature_importance(self, model_name: str = 'model_2') -> pd.DataFrame:
        """Get feature importance for the specified model."""
        return self.trainer.get_feature_importance(model_name)
