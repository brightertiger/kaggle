import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..core.config import Config
from ..models.loss import IOUMetric

class ModelEvaluator:
    """Model evaluation class for Salt Identification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
    def evaluate_fold(self, fold_idx: int, threshold: float = None) -> float:
        """Evaluate a single fold with given threshold"""
        if threshold is None:
            threshold = self.config.IOU_CUTOFF
        
        # Load predictions and ground truth
        scores_path = self.config.SCORES_DIR / "valid" / f"scores_{fold_idx}.npy"
        actuals_path = self.config.SCORES_DIR / "valid" / f"actuals_{fold_idx}.npy"
        
        if not scores_path.exists() or not actuals_path.exists():
            raise FileNotFoundError(f"Predictions not found for fold {fold_idx}")
        
        scores = torch.from_numpy(np.load(scores_path))
        actuals = torch.from_numpy(np.load(actuals_path))
        
        # Create metric function
        metric_fn = IOUMetric(cutoff=threshold, squash=False)
        
        # Calculate metric
        metric = metric_fn(scores, actuals)
        
        return metric
    
    def evaluate_all_folds(self, threshold: float = None) -> Tuple[List[float], float]:
        """Evaluate all folds and return individual and average metrics"""
        if threshold is None:
            threshold = self.config.IOU_CUTOFF
        
        fold_metrics = []
        
        for fold_idx in range(1, self.config.NUM_FOLDS + 1):
            try:
                metric = self.evaluate_fold(fold_idx, threshold)
                fold_metrics.append(metric)
                print(f"Fold {fold_idx}: {metric:.4f}")
            except FileNotFoundError:
                print(f"Predictions not found for fold {fold_idx}")
                continue
        
        if not fold_metrics:
            raise ValueError("No valid predictions found for evaluation")
        
        average_metric = np.mean(fold_metrics)
        print(f"Average metric: {average_metric:.4f}")
        
        return fold_metrics, average_metric
    
    def find_best_threshold(self, threshold_range: Tuple[float, float] = (-0.25, 0.25),
                           step: float = 0.01) -> Tuple[float, float]:
        """Find the best threshold for optimal performance"""
        print("Searching for optimal threshold...")
        
        best_threshold = 0.0
        best_metric = 0.0
        best_fold_metrics = []
        
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        
        for threshold in thresholds:
            try:
                fold_metrics, avg_metric = self.evaluate_all_folds(threshold)
                
                if avg_metric > best_metric:
                    best_metric = avg_metric
                    best_threshold = threshold
                    best_fold_metrics = fold_metrics
                    
            except ValueError:
                continue
        
        print(f"Best threshold: {best_threshold:.3f}")
        print(f"Best metric: {best_metric:.4f}")
        print(f"Individual fold metrics: {[f'{m:.4f}' for m in best_fold_metrics]}")
        
        return best_threshold, best_metric
    
    def evaluate_model_performance(self, model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        print(f"Evaluating model: {model_name}")
        print("=" * 50)
        
        # Find best threshold
        best_threshold, best_metric = self.find_best_threshold()
        
        # Evaluate with best threshold
        fold_metrics, avg_metric = self.evaluate_all_folds(best_threshold)
        
        # Calculate statistics
        std_metric = np.std(fold_metrics)
        min_metric = np.min(fold_metrics)
        max_metric = np.max(fold_metrics)
        
        results = {
            'model_name': model_name,
            'best_threshold': best_threshold,
            'average_metric': avg_metric,
            'std_metric': std_metric,
            'min_metric': min_metric,
            'max_metric': max_metric,
            'fold_metrics': fold_metrics
        }
        
        print("\nEvaluation Results:")
        print(f"Best Threshold: {best_threshold:.3f}")
        print(f"Average Metric: {avg_metric:.4f}")
        print(f"Standard Deviation: {std_metric:.4f}")
        print(f"Min Metric: {min_metric:.4f}")
        print(f"Max Metric: {max_metric:.4f}")
        print(f"Individual Folds: {[f'{m:.4f}' for m in fold_metrics]}")
        
        return results
