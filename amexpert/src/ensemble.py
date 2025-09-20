import pandas as pd
import numpy as np


def rank_blend_predictions(score_paths, output_path):
    """Blend predictions using rank averaging."""
    scores = []
    for path in score_paths:
        score = pd.read_csv(path)
        scores.append(score)
    
    correlations = []
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            corr = pd.Series.corr(scores[i]['redemption_status'], scores[j]['redemption_status'])
            correlations.append(corr)
            print(f"Correlation between model {i+1} and model {j+1}: {corr:.4f}")
    
    means = [score['redemption_status'].mean() for score in scores]
    print(f"Prediction means: {means}")
    
    for i, score in enumerate(scores):
        score['redemption_status'] = score['redemption_status'].rank()
    
    blended_score = scores[0]
    for score in scores[1:]:
        blended_score = blended_score.append(score)
    
    blended_score = blended_score.groupby('id').mean().reset_index()
    blended_score.to_csv(output_path, index=False)
    
    print(f"Final blended predictions shape: {blended_score.shape}")
    return blended_score


def weighted_blend_predictions(score_paths, weights, output_path):
    """Blend predictions using weighted averaging."""
    scores = []
    for path in score_paths:
        score = pd.read_csv(path)
        scores.append(score)
    
    blended_score = scores[0].copy()
    blended_score['redemption_status'] = blended_score['redemption_status'] * weights[0]
    
    for i, score in enumerate(scores[1:], 1):
        blended_score['redemption_status'] += score['redemption_status'] * weights[i]
    
    blended_score.to_csv(output_path, index=False)
    return blended_score


def geometric_mean_blend(score_paths, output_path):
    """Blend predictions using geometric mean."""
    scores = []
    for path in score_paths:
        score = pd.read_csv(path)
        scores.append(score)
    
    blended_score = scores[0].copy()
    blended_score['redemption_status'] = np.log(blended_score['redemption_status'] + 1e-8)
    
    for score in scores[1:]:
        blended_score['redemption_status'] += np.log(score['redemption_status'] + 1e-8)
    
    blended_score['redemption_status'] = np.exp(blended_score['redemption_status'] / len(scores))
    blended_score.to_csv(output_path, index=False)
    return blended_score
