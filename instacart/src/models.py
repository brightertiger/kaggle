#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import roc_curve, auc
import pickle
import os

from .config import Config
from .data_utils import calculate_auc_score


class XGBoostModel:
    def __init__(self, config: Config, model_type: str = 'level1'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
        if model_type == 'level1':
            self.params = config.xgb_params.copy()
            self.training_params = config.xgb_training_params.copy()
        elif model_type == 'level2':
            self.params = config.level2_params.copy()
            self.training_params = config.level2_training_params.copy()
        else:
            raise ValueError("model_type must be 'level1' or 'level2'")
    
    def prepare_data(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                    target_col: str = 'reordered') -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        feature_cols = [col for col in train_data.columns if col not in ['user_id', 'product_id', 'eval_set', target_col]]
        
        train_matrix = xgb.DMatrix(
            label=train_data[target_col], 
            data=train_data[feature_cols]
        )
        valid_matrix = xgb.DMatrix(
            label=valid_data[target_col], 
            data=valid_data[feature_cols]
        )
        
        return train_matrix, valid_matrix, feature_cols
    
    def train(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
             target_col: str = 'reordered', save_path: Optional[str] = None) -> Dict[str, Any]:
        train_matrix, valid_matrix, feature_cols = self.prepare_data(train_data, valid_data, target_col)
        
        training_params = self.training_params.copy()
        training_params['params'] = self.params
        training_params['dtrain'] = train_matrix
        training_params['evals'] = [(train_matrix, 'train'), (valid_matrix, 'valid')]
        
        if self.model_type == 'level2':
            training_params['callbacks'] = [
                xgb.callback.reset_learning_rate([0.02] * training_params['num_boost_round'])
            ]
        
        self.model = xgb.train(**training_params)
        
        if save_path:
            self.save_model(save_path)
        
        train_pred = self.model.predict(train_matrix)
        valid_pred = self.model.predict(valid_matrix)
        
        train_auc = calculate_auc_score(train_data[target_col], train_pred)
        valid_auc = calculate_auc_score(valid_data[target_col], valid_pred)
        
        self.feature_importance = self.get_feature_importance(feature_cols)
        
        results = {
            'train_auc': train_auc,
            'valid_auc': valid_auc,
            'feature_importance': self.feature_importance,
            'feature_cols': feature_cols
        }
        
        del train_matrix, valid_matrix
        return results
    
    def predict(self, data: pd.DataFrame, target_col: str = 'reordered') -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        feature_cols = [col for col in data.columns if col not in ['user_id', 'product_id', 'eval_set', target_col]]
        data_matrix = xgb.DMatrix(data=data[feature_cols])
        
        predictions = self.model.predict(data_matrix)
        del data_matrix
        
        return predictions
    
    def get_feature_importance(self, feature_cols: list) -> Dict[str, float]:
        if self.model is None:
            return {}
        
        importance = self.model.get_fscore()
        importance_dict = {feature_cols[i]: importance.get(f'f{i}', 0) for i in range(len(feature_cols))}
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        
        model_info = {
            'model_type': self.model_type,
            'params': self.params,
            'training_params': self.training_params,
            'feature_importance': self.feature_importance
        }
        
        info_path = path.replace('.model', '_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
    
    def load_model(self, path: str):
        self.model = xgb.Booster({'nthread': self.params.get('nthread', 4)})
        self.model.load_model(path)
        
        info_path = path.replace('.model', '_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)
                self.feature_importance = model_info.get('feature_importance', {})


class Level1Ensemble:
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.feature_sets = {}
    
    def add_model(self, name: str, features: list, model_type: str = 'level1'):
        self.models[name] = XGBoostModel(self.config, model_type)
        self.feature_sets[name] = features
    
    def train_all(self, train_data: pd.DataFrame, valid_data: pd.DataFrame):
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name} model...")
            
            features = self.feature_sets[name]
            model_data = train_data[['user_id', 'product_id', 'eval_set', 'reordered'] + features]
            valid_model_data = valid_data[['user_id', 'product_id', 'eval_set', 'reordered'] + features]
            
            model_results = model.train(model_data, valid_model_data)
            results[name] = model_results
            
            print(f"{name} - Train AUC: {model_results['train_auc']:.4f}, Valid AUC: {model_results['valid_auc']:.4f}")
        
        return results
    
    def predict_all(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        
        for name, model in self.models.items():
            features = self.feature_sets[name]
            model_data = data[['user_id', 'product_id', 'eval_set'] + features]
            predictions[name] = model.predict(model_data)
        
        return predictions
    
    def save_all(self, base_path: str):
        for name, model in self.models.items():
            model_path = os.path.join(base_path, f'{name}_model.model')
            model.save_model(model_path)


class Level2Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = XGBoostModel(config, 'level2')
    
    def prepare_level2_data(self, dependent_data: pd.DataFrame, 
                          independent_data: pd.DataFrame) -> pd.DataFrame:
        data = dependent_data.merge(independent_data, on=['user_id', 'product_id', 'eval_set'], how='inner')
        return data
    
    def train(self, dependent_data: pd.DataFrame, independent_data: pd.DataFrame, 
             save_path: Optional[str] = None) -> Dict[str, Any]:
        data = self.prepare_level2_data(dependent_data, independent_data)
        
        train_data = data[data['eval_set'] == 'train']
        valid_data = data[data['eval_set'] == 'valid']
        
        results = self.model.train(train_data, valid_data, save_path=save_path)
        
        del data, train_data, valid_data
        return results
    
    def predict(self, dependent_data: pd.DataFrame, independent_data: pd.DataFrame) -> np.ndarray:
        data = self.prepare_level2_data(dependent_data, independent_data)
        predictions = self.model.predict(data)
        
        del data
        return predictions
    
    def save_model(self, path: str):
        self.model.save_model(path)
    
    def load_model(self, path: str):
        self.model.load_model(path)


class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      threshold: float = 0.5) -> Dict[str, float]:
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        y_pred_binary = (y_pred > threshold).astype(int)
        
        metrics = {
            'auc': calculate_auc_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'f1_score': f1_score(y_true, y_pred_binary)
        }
        
        return metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             metric: str = 'f1') -> Tuple[float, float]:
        from sklearn.metrics import precision_recall_curve, f1_score
        
        if metric == 'f1':
            thresholds = np.arange(0.1, 0.9, 0.01)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred_binary = (y_pred > threshold).astype(int)
                f1_scores.append(f1_score(y_true, y_pred_binary))
            
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_score = f1_scores[best_idx]
            
            return best_threshold, best_score
        
        elif metric == 'precision_recall':
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = f1_scores[best_idx]
            
            return best_threshold, best_score
        
        else:
            raise ValueError("metric must be 'f1' or 'precision_recall'")


class SubmissionGenerator:
    def __init__(self, config: Config):
        self.config = config
    
    def create_submission(self, predictions: np.ndarray, user_ids: np.ndarray, 
                         product_ids: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        y_pred_binary = (predictions > threshold).astype(int)
        
        submission = pd.DataFrame({
            'user_id': user_ids,
            'product_id': product_ids,
            'reordered': y_pred_binary
        })
        
        submission = submission[submission['reordered'] == 1]
        
        submission_grouped = submission.groupby('user_id')['product_id'].apply(
            lambda x: ' '.join(map(str, x))
        ).reset_index()
        
        submission_grouped.columns = ['order_id', 'products']
        
        all_users = pd.DataFrame({'order_id': user_ids})
        final_submission = all_users.merge(submission_grouped, on='order_id', how='left')
        final_submission['products'] = final_submission['products'].fillna('None')
        
        return final_submission
    
    def save_submission(self, submission: pd.DataFrame, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        submission.to_csv(path, index=False)
