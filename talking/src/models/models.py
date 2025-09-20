import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TalkingDataModel:
    """LightGBM model wrapper for TalkingData AdTracking fraud detection."""
    
    def __init__(self, config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.params = self.config.LGB_PARAMS.get(model_name, {})
        
    def train(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
              train_labels: np.ndarray, valid_labels: np.ndarray) -> None:
        """Train LightGBM model."""
        
        # Prepare data
        train_features = self._prepare_features(train_data)
        valid_features = self._prepare_features(valid_data)
        
        # Create LightGBM datasets
        train_dataset = self._create_dataset(train_features, train_labels)
        valid_dataset = self._create_dataset(valid_features, valid_labels)
        
        # Training parameters
        train_params = {
            'params': self.params,
            'train_set': train_dataset,
            'valid_sets': [valid_dataset],
            'num_boost_round': 1000,
            'early_stopping_rounds': self.config.EARLY_STOPPING_ROUNDS,
            'verbose_eval': 50
        }
        
        # Train model
        self.model = lgb.train(**train_params)
        
        # Save model
        model_path = self.config.MODELS_DIR / f'{self.model_name}.model'
        self.model.save_model(str(model_path))
        
        print(f"Model {self.model_name} trained and saved")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            # Load model if not already loaded
            model_path = self.config.MODELS_DIR / f'{self.model_name}.model'
            self.model = lgb.Booster(model_file=str(model_path))
        
        features = self._prepare_features(data)
        predictions = self.model.predict(features)
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            model_path = self.config.MODELS_DIR / f'{self.model_name}.model'
            self.model = lgb.Booster(model_file=str(model_path))
        
        importance = self.model.feature_importance(importance_type=importance_type)
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': importance
        })
        importance_df['importance'] = importance_df['importance'] / importance_df['importance'].max()
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction."""
        features = data.drop(['is_attributed', 'day', 'click_id'], axis=1, errors='ignore')
        return features
    
    def _create_dataset(self, features: pd.DataFrame, labels: np.ndarray = None) -> lgb.Dataset:
        """Create LightGBM dataset."""
        params = {
            'feature_name': list(features.columns),
            'categorical_feature': self.config.CATEGORICAL_FEATURES
        }
        
        if labels is not None:
            params['label'] = labels
        
        return lgb.Dataset(features.values, **params)

class ModelEnsemble:
    """Model ensemble for combining multiple model predictions."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = self.config.ENSEMBLE_WEIGHTS
        
    def load_models(self, model_names: List[str]):
        """Load trained models."""
        for model_name in model_names:
            self.models[model_name] = TalkingDataModel(self.config, model_name)
    
    def predict_ensemble(self, data: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(data)
            weight = self.weights.get(f'score_{model_name.split("_")[-1]}', 1.0)
            predictions[model_name] = pred * weight
        
        # Combine predictions
        ensemble_pred = np.zeros(len(data))
        for pred in predictions.values():
            ensemble_pred += pred
        
        # Normalize
        ensemble_pred = ensemble_pred / ensemble_pred.max()
        
        return ensemble_pred
    
    def blend_predictions(self, prediction_files: List[str], 
                         output_file: str, mode: str = 'submit') -> pd.DataFrame:
        """Blend predictions from multiple models."""
        
        # Load predictions
        predictions = {}
        for i, file_path in enumerate(prediction_files, 1):
            score_name = f'score_{i}'
            pred_df = pd.read_csv(file_path)
            
            if mode == 'submit':
                pred_df = pred_df.rename(columns={'is_attributed': score_name})
            else:
                pred_df = pred_df.rename(columns={'score': score_name})
            
            predictions[score_name] = pred_df
        
        # Merge all predictions
        result = predictions['score_1'].copy()
        for score_name, pred_df in predictions.items():
            if score_name != 'score_1':
                if mode == 'submit':
                    result = result.merge(pred_df, on='click_id')
                else:
                    result = result.merge(pred_df, on=['click_id', 'is_attributed'])
        
        # Calculate weighted ensemble
        ensemble_cols = [f'score_{i}' for i in range(1, len(predictions) + 1)]
        result['is_attributed'] = 0
        
        for i, col in enumerate(ensemble_cols, 1):
            weight = self.weights.get(col, 1.0)
            result['is_attributed'] += weight * result[col]
        
        # Normalize
        result['is_attributed'] = result['is_attributed'] / result['is_attributed'].max()
        
        # Save result
        if mode == 'submit':
            result[['click_id', 'is_attributed']].to_csv(output_file, index=False)
        else:
            result[['click_id', 'is_attributed']].to_csv(output_file, index=False)
        
        return result[['click_id', 'is_attributed']]
