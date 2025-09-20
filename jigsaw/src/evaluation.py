#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn import metrics

from .config import Config


class BiasEvaluator:
    """Evaluation utilities for bias metrics in toxic comment classification."""
    
    def __init__(self, config: Config):
        self.config = config
        self.identity_columns = config.identity_columns
        self.eval_config = config.eval_config
    
    def compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC score with error handling."""
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.0
    
    def calculate_overall_auc(self, df: pd.DataFrame, model_name: str = 'prediction') -> float:
        """Calculate overall AUC score."""
        true_labels = df[self.config.aux_labels[0]] > self.eval_config['toxicity_threshold']
        predicted_labels = df[model_name]
        return self.compute_auc(true_labels, predicted_labels)
    
    def power_mean(self, series: pd.Series, p: float) -> float:
        """Calculate power mean of a series."""
        if len(series) == 0:
            return 0.0
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)
    
    def compute_subgroup_auc(self, df: pd.DataFrame, subgroup: str, 
                           label: str = 'target', model_name: str = 'prediction') -> float:
        """Compute AUC for a specific subgroup."""
        subgroup_examples = df[df[subgroup] > self.eval_config['identity_threshold']]
        if len(subgroup_examples) == 0:
            return 0.0
        return self.compute_auc(
            (subgroup_examples[label] > self.eval_config['toxicity_threshold']),
            subgroup_examples[model_name]
        )
    
    def compute_bpsn_auc(self, df: pd.DataFrame, subgroup: str, 
                        label: str = 'target', model_name: str = 'prediction') -> float:
        """Compute Background Positive, Subgroup Negative (BPSN) AUC."""
        subgroup_negative = df[
            (df[subgroup] > self.eval_config['identity_threshold']) & 
            (df[label] <= self.eval_config['toxicity_threshold'])
        ]
        non_subgroup_positive = df[
            (df[subgroup] <= self.eval_config['identity_threshold']) & 
            (df[label] > self.eval_config['toxicity_threshold'])
        ]
        
        if len(subgroup_negative) == 0 or len(non_subgroup_positive) == 0:
            return 0.0
        
        examples = pd.concat([subgroup_negative, non_subgroup_positive])
        return self.compute_auc(examples[label] > self.eval_config['toxicity_threshold'], 
                               examples[model_name])
    
    def compute_bnsp_auc(self, df: pd.DataFrame, subgroup: str, 
                        label: str = 'target', model_name: str = 'prediction') -> float:
        """Compute Background Negative, Subgroup Positive (BNSP) AUC."""
        subgroup_positive = df[
            (df[subgroup] > self.eval_config['identity_threshold']) & 
            (df[label] > self.eval_config['toxicity_threshold'])
        ]
        non_subgroup_negative = df[
            (df[subgroup] <= self.eval_config['identity_threshold']) & 
            (df[label] <= self.eval_config['toxicity_threshold'])
        ]
        
        if len(subgroup_positive) == 0 or len(non_subgroup_negative) == 0:
            return 0.0
        
        examples = pd.concat([subgroup_positive, non_subgroup_negative])
        return self.compute_auc(examples[label] > self.eval_config['toxicity_threshold'], 
                               examples[model_name])
    
    def compute_bias_metrics_for_model(self, dataset: pd.DataFrame, 
                                     model: str = 'prediction', 
                                     label_col: str = 'target') -> pd.DataFrame:
        """Compute bias metrics for all identity groups."""
        records = []
        
        for subgroup in self.identity_columns:
            if subgroup not in dataset.columns:
                continue
                
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[dataset[subgroup] > self.eval_config['identity_threshold']])
            }
            
            record['subgroup_auc'] = self.compute_subgroup_auc(dataset, subgroup, label_col, model)
            record['bpsn_auc'] = self.compute_bpsn_auc(dataset, subgroup, label_col, model)
            record['bnsp_auc'] = self.compute_bnsp_auc(dataset, subgroup, label_col, model)
            
            records.append(record)
        
        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
    
    def get_final_metric(self, bias_df: pd.DataFrame, overall_auc: float) -> float:
        """Calculate the final bias-aware metric."""
        power = self.eval_config['subgroup_auc_weight']
        
        bias_score = np.average([
            self.power_mean(bias_df['subgroup_auc'], power),
            self.power_mean(bias_df['bpsn_auc'], power),
            self.power_mean(bias_df['bnsp_auc'], power)
        ])
        
        overall_weight = self.eval_config['overall_model_weight']
        final_score = (overall_weight * overall_auc) + ((1 - overall_weight) * bias_score)
        
        return final_score
    
    def evaluate_model(self, predictions: pd.DataFrame, 
                      ground_truth: pd.DataFrame, 
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        merged_data = predictions.merge(ground_truth, on='id', how='inner')
        
        overall_auc = self.calculate_overall_auc(merged_data)
        bias_metrics = self.compute_bias_metrics_for_model(merged_data)
        final_metric = self.get_final_metric(bias_metrics, overall_auc)
        
        results = {
            'overall_auc': overall_auc,
            'final_metric': final_metric,
            'bias_metrics': bias_metrics,
            'subgroup_auc_mean': bias_metrics['subgroup_auc'].mean(),
            'bpsn_auc_mean': bias_metrics['bpsn_auc'].mean(),
            'bnsp_auc_mean': bias_metrics['bnsp_auc'].mean()
        }
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            predictions.to_csv(f"{save_path}_predictions.csv", index=False)
            bias_metrics.to_csv(f"{save_path}_bias_metrics.csv", index=False)
            
            summary_df = pd.DataFrame([{
                'metric': 'overall_auc',
                'value': overall_auc
            }, {
                'metric': 'final_metric',
                'value': final_metric
            }])
            summary_df.to_csv(f"{save_path}_summary.csv", index=False)
        
        return results


class ModelEvaluator:
    """General model evaluation utilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.bias_evaluator = BiasEvaluator(config)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         threshold: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        y_pred_binary = (y_pred > threshold).astype(int)
        
        metrics_dict = {
            'auc': metrics.roc_auc_score(y_true, y_pred),
            'accuracy': metrics.accuracy_score(y_true, y_pred_binary),
            'precision': metrics.precision_score(y_true, y_pred_binary),
            'recall': metrics.recall_score(y_true, y_pred_binary),
            'f1_score': metrics.f1_score(y_true, y_pred_binary),
            'log_loss': metrics.log_loss(y_true, y_pred)
        }
        
        return metrics_dict
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              metric: str = 'f1') -> Tuple[float, float]:
        """Find optimal threshold for a given metric."""
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            
            if metric == 'f1':
                score = metrics.f1_score(y_true, y_pred_binary)
            elif metric == 'precision':
                score = metrics.precision_score(y_true, y_pred_binary)
            elif metric == 'recall':
                score = metrics.recall_score(y_true, y_pred_binary)
            elif metric == 'accuracy':
                score = metrics.accuracy_score(y_true, y_pred_binary)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return optimal_threshold, optimal_score
    
    def evaluate_predictions(self, predictions: pd.DataFrame, 
                           ground_truth: pd.DataFrame,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate predictions comprehensively."""
        merged_data = predictions.merge(ground_truth, on='id', how='inner')
        
        y_true = merged_data['target'].values
        y_pred = merged_data['prediction'].values
        
        basic_metrics = self.calculate_metrics(y_true, y_pred)
        optimal_threshold, optimal_f1 = self.find_optimal_threshold(y_true, y_pred)
        
        bias_results = self.bias_evaluator.evaluate_model(
            predictions, ground_truth, save_path
        )
        
        results = {
            **basic_metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            **bias_results
        }
        
        return results
