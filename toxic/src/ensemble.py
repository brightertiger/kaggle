import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from src.config import Config


class EnsembleModel:
    """Ensemble methods for combining multiple model predictions"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def simple_averaging(self, predictions: List[pd.DataFrame], 
                        target_columns: List[str]) -> pd.DataFrame:
        """Simple averaging of predictions"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Start with first prediction
        result = predictions[0].copy()
        
        # Average across all predictions
        for col in target_columns:
            if col in result.columns:
                all_values = [pred[col].values for pred in predictions if col in pred.columns]
                if all_values:
                    result[col] = np.mean(all_values, axis=0)
        
        return result
    
    def weighted_averaging(self, predictions: List[pd.DataFrame], 
                          weights: List[float], 
                          target_columns: List[str]) -> pd.DataFrame:
        """Weighted averaging of predictions"""
        if len(predictions) != len(weights):
            raise ValueError("Number of predictions must match number of weights")
        
        result = predictions[0].copy()
        
        for col in target_columns:
            if col in result.columns:
                weighted_sum = np.zeros(len(result))
                total_weight = 0
                
                for pred, weight in zip(predictions, weights):
                    if col in pred.columns:
                        weighted_sum += pred[col].values * weight
                        total_weight += weight
                
                if total_weight > 0:
                    result[col] = weighted_sum / total_weight
        
        return result
    
    def stacking(self, train_predictions: List[pd.DataFrame], 
                 test_predictions: List[pd.DataFrame],
                 train_labels: pd.DataFrame,
                 target_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stacking ensemble with logistic regression meta-learner"""
        
        # Prepare training features for meta-learner
        train_features = self._prepare_stacking_features(train_predictions, target_columns)
        
        # Prepare test features
        test_features = self._prepare_stacking_features(test_predictions, target_columns)
        
        train_ids = train_predictions[0][['id']].copy()
        test_ids = test_predictions[0][['id']].copy()
        
        train_results = train_ids.copy()
        test_results = test_ids.copy()
        
        # Train meta-learner for each target
        for col in target_columns:
            if col in train_labels.columns:
                meta_model = LogisticRegression(C=1.0, random_state=42)
                meta_model.fit(train_features, train_labels[col])
                
                train_results[col] = meta_model.predict_proba(train_features)[:, 1]
                test_results[col] = meta_model.predict_proba(test_features)[:, 1]
        
        return train_results, test_results
    
    def _prepare_stacking_features(self, predictions: List[pd.DataFrame], 
                                 target_columns: List[str]) -> np.ndarray:
        """Prepare features for stacking"""
        features = []
        
        for pred in predictions:
            for col in target_columns:
                if col in pred.columns:
                    features.append(pred[col].values)
        
        return np.column_stack(features) if features else np.array([]).reshape(len(predictions[0]), 0)
    
    def blend_predictions(self, predictions: Dict[str, pd.DataFrame], 
                         weights: Dict[str, float],
                         target_columns: List[str]) -> pd.DataFrame:
        """Blend predictions from different models with custom weights"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Initialize result with first prediction
        result = list(predictions.values())[0].copy()
        
        for col in target_columns:
            if col in result.columns:
                weighted_sum = np.zeros(len(result))
                total_weight = 0
                
                for model_name, pred in predictions.items():
                    if col in pred.columns:
                        weight = weights.get(model_name, 1.0)
                        weighted_sum += pred[col].values * weight
                        total_weight += weight
                
                if total_weight > 0:
                    result[col] = weighted_sum / total_weight
        
        return result


class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def evaluate_single_model(self, predictions: pd.DataFrame, 
                            actual: pd.DataFrame,
                            target_columns: List[str]) -> Dict[str, float]:
        """Evaluate a single model's performance"""
        results = {}
        overall = 0.0
        
        for col in target_columns:
            if col in predictions.columns and col in actual.columns:
                fpr, tpr, _ = roc_curve(actual[col], predictions[col])
                auc_score = auc(fpr, tpr)
                results[col] = round(auc_score, 4)
                overall += auc_score
        
        results['overall'] = round(overall / len(target_columns), 4)
        return results
    
    def evaluate_ensemble(self, predictions: Dict[str, pd.DataFrame],
                         actual: pd.DataFrame,
                         target_columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models and ensemble"""
        results = {}
        
        for model_name, pred in predictions.items():
            results[model_name] = self.evaluate_single_model(pred, actual, target_columns)
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, Dict[str, float]]):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Get all target columns
        target_columns = [col for col in self.config.evaluation.target_columns 
                         if col != 'overall']
        
        # Print header
        header = f"{'Model':<20}"
        for col in target_columns:
            header += f"{col:<12}"
        header += "Overall"
        print(header)
        print("-" * len(header))
        
        # Print results for each model
        for model_name, scores in results.items():
            row = f"{model_name:<20}"
            for col in target_columns:
                score = scores.get(col, 0.0)
                row += f"{score:<12.4f}"
            row += f"{scores.get('overall', 0.0):.4f}"
            print(row)
        
        print("="*60)
    
    def find_best_model(self, results: Dict[str, Dict[str, float]]) -> str:
        """Find the best performing model based on overall score"""
        best_model = None
        best_score = 0.0
        
        for model_name, scores in results.items():
            overall_score = scores.get('overall', 0.0)
            if overall_score > best_score:
                best_score = overall_score
                best_model = model_name
        
        return best_model, best_score
