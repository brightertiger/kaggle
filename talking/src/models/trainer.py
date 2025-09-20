import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

from .models import TalkingDataModel, ModelEnsemble

class ModelTrainer:
    """Model trainer for TalkingData AdTracking fraud detection."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        
    def train_single_model(self, model_name: str, train_data: pd.DataFrame, 
                          valid_data: pd.DataFrame) -> Dict:
        """Train a single model and return evaluation metrics."""
        
        print(f"Training model: {model_name}")
        
        # Prepare labels
        train_labels = train_data[self.config.TARGET_COLUMN].values
        valid_labels = valid_data[self.config.TARGET_COLUMN].values
        
        # Create and train model
        model = TalkingDataModel(self.config, model_name)
        model.train(train_data, valid_data, train_labels, valid_labels)
        
        # Make predictions
        train_pred = model.predict(train_data)
        valid_pred = model.predict(valid_data)
        
        # Calculate metrics
        train_auc = roc_auc_score(train_labels, train_pred)
        valid_auc = roc_auc_score(valid_labels, valid_pred)
        
        # Store model
        self.models[model_name] = model
        
        metrics = {
            'model_name': model_name,
            'train_auc': train_auc,
            'valid_auc': valid_auc,
            'train_samples': len(train_data),
            'valid_samples': len(valid_data),
            'positive_rate_train': train_labels.mean(),
            'positive_rate_valid': valid_labels.mean()
        }
        
        print(f"Model {model_name} - Train AUC: {train_auc:.4f}, Valid AUC: {valid_auc:.4f}")
        
        return metrics
    
    def train_multiple_models(self, train_data: pd.DataFrame, 
                             valid_data: pd.DataFrame, 
                             model_names: List[str]) -> List[Dict]:
        """Train multiple models and return evaluation metrics."""
        
        all_metrics = []
        
        for model_name in model_names:
            metrics = self.train_single_model(model_name, train_data, valid_data)
            all_metrics.append(metrics)
        
        return all_metrics
    
    def generate_predictions(self, data: pd.DataFrame, 
                           model_name: str, 
                           output_file: Optional[str] = None) -> pd.DataFrame:
        """Generate predictions using a trained model."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        model = self.models[model_name]
        predictions = model.predict(data)
        
        # Create result dataframe
        result = pd.DataFrame({
            'click_id': data['click_id'].values,
            'score': predictions
        })
        
        # Add target if available
        if self.config.TARGET_COLUMN in data.columns:
            result['is_attributed'] = data[self.config.TARGET_COLUMN].values
        
        # Save if output file specified
        if output_file:
            result.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        
        return result
    
    def evaluate_model(self, valid_data: pd.DataFrame, model_name: str) -> Dict:
        """Evaluate model performance on validation data."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        predictions = model.predict(valid_data)
        true_labels = valid_data[self.config.TARGET_COLUMN].values
        
        # Calculate AUC
        auc_score = roc_auc_score(true_labels, predictions)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        
        # Find best threshold (Youden's J statistic)
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        evaluation = {
            'model_name': model_name,
            'auc_score': auc_score,
            'best_threshold': best_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        print(f"Model {model_name} - AUC: {auc_score:.4f}, Best Threshold: {best_threshold:.4f}")
        
        return evaluation
    
    def get_feature_importance(self, model_name: str, 
                              top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a trained model."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        importance_df = model.get_feature_importance()
        
        return importance_df.head(top_n)
    
    def save_feature_importance(self, model_name: str, output_file: str):
        """Save feature importance to CSV file."""
        importance_df = self.get_feature_importance(model_name)
        importance_df.to_csv(output_file, index=False)
        print(f"Feature importance saved to {output_file}")
    
    def create_submission(self, test_data: pd.DataFrame, 
                         model_name: str, 
                         output_file: str) -> pd.DataFrame:
        """Create submission file for test data."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        predictions = model.predict(test_data)
        
        submission = pd.DataFrame({
            'click_id': test_data['click_id'].values,
            'is_attributed': predictions
        })
        
        submission.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        
        return submission
