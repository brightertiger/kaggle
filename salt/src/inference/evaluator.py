"""Native-resolution out-of-fold scoring and logit threshold selection."""
import numpy as np
import torch
from ..models.loss import IOUMetric


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def _load_fold(self, fold):
        directory = self.config.SCORES_DIR / 'valid'
        scores = torch.from_numpy(np.load(directory / f'scores_{fold}.npy'))
        actuals = torch.from_numpy(np.load(directory / f'actuals_{fold}.npy'))
        if scores.shape != actuals.shape or not torch.isfinite(scores).all():
            raise ValueError(f'Invalid validation predictions for fold {fold}')
        return scores, actuals

    def _metric(self, scores, actuals, threshold):
        return IOUMetric(cutoff=threshold, squash=False,
                         min_salt_pixels=self.config.MIN_SALT_PIXELS)(scores, actuals)

    def evaluate_fold(self, fold_idx, threshold=None):
        threshold = self.config.IOU_CUTOFF if threshold is None else threshold
        return self._metric(*self._load_fold(fold_idx), threshold)

    def evaluate_all_folds(self, threshold=None):
        threshold = self.config.IOU_CUTOFF if threshold is None else threshold
        folds = [self._load_fold(i) for i in range(1, self.config.NUM_FOLDS + 1)]
        metrics = [self._metric(scores, actuals, threshold) for scores, actuals in folds]
        return metrics, float(np.average(metrics, weights=[len(scores) for scores, _ in folds]))

    def find_best_threshold(self, threshold_range=(-0.25, 0.25), step=0.01):
        if step <= 0 or threshold_range[0] > threshold_range[1]:
            raise ValueError('Invalid threshold range or step')
        # Read once, rather than reading every fold again for every candidate.
        try:
            folds = [self._load_fold(i) for i in range(1, self.config.NUM_FOLDS + 1)]
        except FileNotFoundError as error:
            raise ValueError('Predictions for every fold are required; run predict first') from error
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step / 2, step)
        best_threshold, best_metric = None, -float('inf')
        for threshold in thresholds:
            metrics = [self._metric(scores, actuals, threshold) for scores, actuals in folds]
            metric = float(np.average(metrics, weights=[len(scores) for scores, _ in folds]))
            if metric > best_metric:
                best_threshold, best_metric = float(threshold), metric
        print(f'Best logit threshold: {best_threshold:.3f}; OOF metric: {best_metric:.4f}')
        return best_threshold, best_metric

    def evaluate_model_performance(self, model_name):
        self.config.MODEL_NAME = model_name
        threshold, metric = self.find_best_threshold()
        folds, _ = self.evaluate_all_folds(threshold)
        return {'model_name': model_name, 'best_threshold': threshold,
                'average_metric': metric, 'std_metric': float(np.std(folds)),
                'min_metric': min(folds), 'max_metric': max(folds), 'fold_metrics': folds}
