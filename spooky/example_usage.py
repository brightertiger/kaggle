#!/usr/bin/env python3
"""Small API examples. Run the complete synthetic example with no arguments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.feature_engineering import NaiveBayesFeatureEngineer, TextFeatureEngineer
from src.models import NaiveBayesModel, XGBoostModel
from src.pipeline import SpookyAuthorPipeline


def example_text_feature_extraction(config=None):
    frame = pd.DataFrame({'text': ['A shadow moved beyond the window.', 'The chamber was silent.']})
    return TextFeatureEngineer(config).extract_all_features(frame)


def example_naive_bayes_features():
    train = ['A shadow moved beyond the window.', 'The chamber was silent.']
    test = ['Ancient ruins haunted my dreams.']
    engineer = NaiveBayesFeatureEngineer()
    return engineer.fit_transform_svd_features(train, test)


def example_model_training():
    config = Config()
    config.XGB_NUM_ROUNDS = 3
    rng = np.random.default_rng(config.RANDOM_STATE)
    # MultinomialNB requires nonnegative counts.
    features = rng.poisson(2, size=(30, 8))
    targets = np.tile(np.arange(config.NUM_CLASSES), 10)
    test_features = rng.poisson(2, size=(6, 8))
    nb_train, nb_test = NaiveBayesModel(config).train_cv(features, targets, test_features)
    train = pd.DataFrame(features)
    train['author'] = targets
    model = XGBoostModel(config=config)
    model.train(train)
    return nb_train, nb_test, model.predict(pd.DataFrame(test_features))


def example_full_pipeline(data_dir='data', glove_path=Config.GLOVE_PATH):
    config = Config()
    config.GLOVE_PATH = Path(glove_path)
    pipeline = SpookyAuthorPipeline(data_dir=data_dir, config=config)
    cv_history, predictions = pipeline.run_full_pipeline()
    print(predictions.head())
    print('XGBoost CV history (not nested stack validation):', cv_history)
    print('Feature importance:', pipeline.get_feature_importance()[:10])
    return cv_history, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', help='Use real train.csv/test.csv instead of synthetic data')
    parser.add_argument('--glove_path', default=str(Config.GLOVE_PATH))
    args = parser.parse_args()
    if args.data_dir:
        example_full_pipeline(args.data_dir, args.glove_path)
    else:
        from dry_run import main as dry_run
        dry_run()


if __name__ == '__main__':
    main()
