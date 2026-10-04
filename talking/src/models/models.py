import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import List

class TalkingDataModel:
    """LightGBM model wrapper for TalkingData AdTracking fraud detection."""
    
    def __init__(self, config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        if model_name not in self.config.LGB_PARAMS:
            raise ValueError(f"No LightGBM parameters configured for {model_name}")
        self.params = dict(self.config.LGB_PARAMS[model_name])
        self.params.update(num_threads=config.NUM_THREADS, seed=config.RANDOM_STATE)
        
    def train(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
              train_labels: np.ndarray, valid_labels: np.ndarray) -> None:
        """Train LightGBM model."""
        
        # Prepare data
        train_features = self._prepare_features(train_data)
        valid_features = self._prepare_features(valid_data)
        
        # Create LightGBM datasets
        train_dataset = self._create_dataset(train_features, train_labels)
        valid_dataset = self._create_dataset(valid_features, valid_labels, reference=train_dataset)
        
        # Training parameters
        train_params = {
            'params': self.params,
            'train_set': train_dataset,
            'valid_sets': [valid_dataset],
            'num_boost_round': self.config.NUM_BOOST_ROUND,
            'callbacks': [lgb.log_evaluation(self.config.LOG_EVALUATION_PERIOD)]
        }
        if self.config.EARLY_STOPPING_ROUNDS > 0:
            train_params['callbacks'].append(lgb.early_stopping(
                self.config.EARLY_STOPPING_ROUNDS,
                verbose=self.config.LOG_EVALUATION_PERIOD > 0,
            ))
        
        # Train model
        self.model = lgb.train(**train_params)
        
        # Save model
        model_path = self.config.MODELS_DIR / f'{self.model_name}.model'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        
        print(f"Model {self.model_name} trained and saved")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            # Load model if not already loaded
            model_path = self.config.MODELS_DIR / f'{self.model_name}.model'
            self.model = lgb.Booster(model_file=str(model_path))
        
        features = self._prepare_features(data)
        predictions = self.model.predict(features, num_threads=self.config.NUM_THREADS)
        
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
        maximum = importance_df['importance'].max()
        if maximum > 0:
            importance_df['importance'] = importance_df['importance'] / maximum
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction."""
        features = data.drop(
            [self.config.TARGET_COLUMN, 'day', 'click_id', 'click_time', 'attributed_time'],
            axis=1, errors='ignore',
        )
        if self.model is not None:
            # Persisted feature names determine prediction order across processes.
            features = features.loc[:, self.model.feature_name()]
        return features
    
    def _create_dataset(self, features: pd.DataFrame, labels: np.ndarray = None,
                        reference=None) -> lgb.Dataset:
        """Create LightGBM dataset."""
        params = {
            'feature_name': list(features.columns),
            'categorical_feature': [name for name in self.config.CATEGORICAL_FEATURES
                                    if name in features.columns],
            'reference': reference,
        }
        
        if labels is not None:
            params['label'] = labels
        
        return lgb.Dataset(features, **params)

class ModelEnsemble:
    """Model ensemble for combining multiple model predictions."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = self.config.ENSEMBLE_WEIGHTS
        
    def load_models(self, model_names: List[str]):
        """Load trained models."""
        self.models = {}
        for model_name in model_names:
            self.models[model_name] = TalkingDataModel(self.config, model_name)
    
    def predict_ensemble(self, data: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("Load at least one model before predicting an ensemble")
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
        ensemble_pred = self._normalize(ensemble_pred)
        
        return ensemble_pred
    
    @staticmethod
    def _normalize(predictions):
        """Retain the original AUC-oriented maximum normalization, including zeros."""
        predictions = np.asarray(predictions, dtype=float)
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("Ensemble scores must be finite and nonnegative")
        maximum = predictions.max() if predictions.size else 0
        return predictions / maximum if maximum > 0 else predictions

    def blend_predictions(self, prediction_files: List[str],
                         output_file: str, mode: str = 'submit',
                         model_names: List[str] = None) -> pd.DataFrame:
        """Blend predictions from multiple models."""
        
        if mode not in {'submit', 'score'}:
            raise ValueError("mode must be 'submit' or 'score'")
        if not prediction_files:
            raise ValueError("At least one prediction file is required")
        if model_names is None:
            model_names = [f'model_{i}' for i in range(1, len(prediction_files) + 1)]
        if len(model_names) != len(prediction_files):
            raise ValueError("Provide one model name per prediction file")

        result = None
        scores = None
        prediction_column = 'is_attributed' if mode == 'submit' else 'score'
        for model_name, file_path in zip(model_names, prediction_files):
            frame = pd.read_csv(file_path)
            if frame['click_id'].isna().any() or frame['click_id'].duplicated().any():
                raise ValueError(f"Missing or duplicate click_id in {file_path}")
            frame = frame.set_index('click_id')
            if result is None:
                result = frame.copy()
                scores = np.zeros(len(frame))
            else:
                if len(frame) != len(result) or not result.index.isin(frame.index).all():
                    raise ValueError(f"Mismatched click IDs in {file_path}")
                frame = frame.reindex(result.index)
                if mode == 'score' and not frame['is_attributed'].equals(result['is_attributed']):
                    raise ValueError(f"Mismatched validation labels in {file_path}")
            values = frame[prediction_column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid predictions in {file_path}")
            weight = self.weights.get(f'score_{model_name.split("_")[-1]}', 1.0)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Ensemble weights must be finite and nonnegative")
            scores += weight * values

        result[prediction_column] = self._normalize(scores)
        columns = ['is_attributed'] if mode == 'submit' else ['is_attributed', 'score']
        result = result[columns].reset_index()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        return result
