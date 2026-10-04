#!/usr/bin/env python3
"""Small API examples; training writes only synthetic sample/output directories."""

import pandas as pd

from src.config import get_config
from src.data_utils import TextPreprocessor
from src.ensemble import EnsembleModel


def example_data_preprocessing():
    samples = pd.Series(['Thanks for the edit!', 'WOW!!! #TalkPage :-) https://example.org', None])
    for method in get_config().data.preprocessing_methods:
        print(method, TextPreprocessor.apply_preprocessing(samples, method).tolist())


def example_ensemble_methods():
    config = get_config()
    targets = config.evaluation.target_columns
    first = pd.DataFrame({'id': ['comment_a', 'comment_b']})
    first[targets] = [[0.1] * len(targets), [0.8] * len(targets)]
    second = first.iloc[::-1].copy()  # Blending must align IDs, not row positions.
    second[targets] *= 0.9
    result = EnsembleModel(config).weighted_averaging([first, second], [0.5, 0.5], targets)
    print(result.to_string(index=False))


def example_model_training():
    from dry_run import main as run_synthetic_pipeline
    run_synthetic_pipeline()


def main():
    example_data_preprocessing()
    example_ensemble_methods()
    example_model_training()


if __name__ == '__main__':
    main()
