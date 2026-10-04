"""ID-aligned blending of coupon redemption predictions."""
import numpy as np
import pandas as pd


def _load_aligned_scores(score_paths):
    if not score_paths:
        raise ValueError('At least one prediction file is required')
    scores = []
    for path in score_paths:
        score = pd.read_csv(path)[['id', 'redemption_status']]
        if score['id'].isna().any() or score['id'].duplicated().any():
            raise ValueError(f'{path}: prediction IDs must be non-null and unique')
        if score.empty or not np.isfinite(score['redemption_status']).all():
            raise ValueError(f'{path}: predictions must be nonempty and finite')
        score = score.set_index('id')
        if scores:
            if set(score.index) != set(scores[0].index):
                raise ValueError('All prediction files must have the same IDs')
            score = score.reindex(scores[0].index)
        scores.append(score)
    return scores


def _save(scores, predictions, output_path):
    result = pd.DataFrame({'id': scores[0].index, 'redemption_status': predictions})
    result.to_csv(output_path, index=False)
    return result


def rank_blend_predictions(score_paths, output_path):
    """Average percentile ranks; these scores are not calibrated probabilities."""
    scores = _load_aligned_scores(score_paths)
    ranked = [score['redemption_status'].rank(pct=True).to_numpy() for score in scores]
    return _save(scores, np.mean(ranked, axis=0), output_path)


def weighted_blend_predictions(score_paths, weights, output_path):
    """Average probabilities with nonnegative weights, aligning by ID."""
    scores = _load_aligned_scores(score_paths)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(scores),) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Provide one nonnegative weight per model with a positive sum')
    values = [score['redemption_status'].to_numpy() for score in scores]
    return _save(scores, np.average(values, axis=0, weights=weights), output_path)


def geometric_mean_blend(score_paths, output_path):
    """Geometric mean of nonnegative predictions, aligning by ID."""
    scores = _load_aligned_scores(score_paths)
    values = np.array([score['redemption_status'].to_numpy() for score in scores])
    if (values < 0).any():
        raise ValueError('Geometric blending requires nonnegative predictions')
    result = np.exp(np.log(np.maximum(values, 1e-8)).mean(axis=0))
    result[(values == 0).any(axis=0)] = 0
    return _save(scores, result, output_path)
