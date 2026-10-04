"""ID-aligned blending, optional stacking, and competition ROC AUC scoring."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import Config


def align_predictions(frame: pd.DataFrame, ids, target_columns: List[str]) -> pd.DataFrame:
    """Require one complete row per ID and return it in the requested order."""
    missing = set(['id', *target_columns]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing prediction/label columns: {sorted(missing)}')
    ids = pd.Index(ids, name='id')
    if ids.has_duplicates or ids.isna().any() or frame['id'].isna().any() or frame['id'].duplicated().any():
        raise ValueError('IDs must be unique and non-null')
    if len(frame) != len(ids) or set(frame['id']) != set(ids):
        raise ValueError('Prediction and reference IDs do not match')
    result = frame.set_index('id').loc[ids, target_columns].reset_index()
    values = result[target_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Predictions/labels must be finite values between 0 and 1')
    return result


class EnsembleModel:
    """Combine probabilities only after matching their comment IDs."""

    def __init__(self, config: Config):
        self.config = config

    def simple_averaging(self, predictions: List[pd.DataFrame],
                         target_columns: List[str]) -> pd.DataFrame:
        return self.weighted_averaging(predictions, [1.0] * len(predictions), target_columns)

    def weighted_averaging(self, predictions: List[pd.DataFrame], weights: List[float],
                           target_columns: List[str]) -> pd.DataFrame:
        if not predictions or len(predictions) != len(weights):
            raise ValueError('Provide predictions and one weight per model')
        weights = np.asarray(weights, dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Weights must be finite, nonnegative, and sum to a positive value')
        aligned = [align_predictions(pred, predictions[0]['id'], target_columns)
                   for pred in predictions]
        result = aligned[0].copy()
        result[target_columns] = np.average(
            np.stack([pred[target_columns].to_numpy() for pred in aligned]),
            axis=0, weights=weights,
        )
        return result

    def blend_predictions(self, predictions: Dict[str, pd.DataFrame],
                          weights: Dict[str, float], target_columns: List[str]) -> pd.DataFrame:
        return self.weighted_averaging(
            list(predictions.values()), [weights.get(name, 1.0) for name in predictions],
            target_columns,
        )

    def _prepare_stacking_features(self, predictions: List[pd.DataFrame],
                                   target_columns: List[str]) -> np.ndarray:
        if not predictions:
            raise ValueError('No predictions provided')
        return np.column_stack([
            align_predictions(pred, predictions[0]['id'], target_columns)[target_columns].to_numpy()
            for pred in predictions
        ])

    def stacking(self, train_predictions: List[pd.DataFrame],
                 test_predictions: List[pd.DataFrame], train_labels: pd.DataFrame,
                 target_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on base-model OOF predictions; returned train scores are in-sample.

        Evaluating the meta-learner itself requires a separate validation split.
        The train and test lists must contain the same models in the same order.
        """
        if len(train_predictions) != len(test_predictions):
            raise ValueError('Stacking requires matching train/test model lists')
        train_features = self._prepare_stacking_features(train_predictions, target_columns)
        test_features = self._prepare_stacking_features(test_predictions, target_columns)
        train_results = train_predictions[0][['id']].reset_index(drop=True)
        test_results = test_predictions[0][['id']].reset_index(drop=True)
        labels = align_predictions(train_labels, train_results['id'], target_columns)
        for col in target_columns:
            if labels[col].nunique() == 1:
                train_results[col] = test_results[col] = labels[col].iloc[0]
                continue
            meta_model = LogisticRegression(C=1.0, random_state=self.config.data.random_state,
                                            max_iter=1000)
            meta_model.fit(train_features, labels[col])
            train_results[col] = meta_model.predict_proba(train_features)[:, 1]
            test_results[col] = meta_model.predict_proba(test_features)[:, 1]
        return train_results, test_results


class ModelEvaluator:
    """Mean per-label ROC AUC; undefined single-class AUCs stay explicitly NaN."""

    def __init__(self, config: Config):
        self.config = config

    def evaluate_single_model(self, predictions: pd.DataFrame, actual: pd.DataFrame,
                              target_columns: List[str]) -> Dict[str, float]:
        pred = align_predictions(predictions, predictions['id'], target_columns)
        actual = align_predictions(actual, pred['id'], target_columns)
        results = {
            col: float(roc_auc_score(actual[col], pred[col]))
            if actual[col].nunique() == 2 else float('nan')
            for col in target_columns
        }
        # Do not quietly call a partial-label average the competition metric.
        results['overall'] = float(np.mean(list(results.values())))
        return results

    def evaluate_ensemble(self, predictions: Dict[str, pd.DataFrame], actual: pd.DataFrame,
                          target_columns: List[str]) -> Dict[str, Dict[str, float]]:
        return {name: self.evaluate_single_model(pred, actual, target_columns)
                for name, pred in predictions.items()}

    def print_evaluation_results(self, results: Dict[str, Dict[str, float]]):
        print('\nOut-of-fold ROC AUC (NaN means a label has only one class):')
        print(pd.DataFrame.from_dict(results, orient='index').to_string(float_format='%.4f'))

    def find_best_model(self, results: Dict[str, Dict[str, float]]) -> Tuple[str | None, float]:
        candidates = [(name, scores['overall']) for name, scores in results.items()
                      if np.isfinite(scores.get('overall', np.nan))]
        return max(candidates, key=lambda item: item[1]) if candidates else (None, float('nan'))
