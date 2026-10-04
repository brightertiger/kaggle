"""Deterministic Spearman scoring, including constant columns in small samples."""
import numpy as np
from scipy.stats import spearmanr


def label_spearman(labels, predictions):
    if len(labels) < 2 or np.ptp(labels) == 0:
        return float('nan')  # No ranking exists for a constant target.
    if np.ptp(predictions) == 0:
        return 0.0
    return float(spearmanr(labels, predictions).statistic)


def mean_spearman(labels, predictions):
    correlations = [label_spearman(y, p) for y, p in zip(labels.T, predictions.T)]
    finite = [value for value in correlations if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0
