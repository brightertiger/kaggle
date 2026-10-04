#!/usr/bin/env python3
"""Small, explicit examples. Run the ensemble example without competition data."""
import argparse
from pathlib import Path

import numpy as np

from src.config import Config
from src.data_utils import MelanomaDataset, load_metadata
from src.ensemble import EnsemblePredictor
from src.inference import MelanomaInference, load_trained_model
from src.models import MelanomaClassifier, MelanomaClassifierV2
from src.pipeline import MelanomaPipeline


def example_basic_training():
    pipeline = MelanomaPipeline(data_dir='data', model_dir='models', score_dir='scores')
    _, score = pipeline.train_single_fold(fold=0, epochs=5)
    print(f'Single fold AUC: {score:.4f}')


def example_cross_validation():
    pipeline = MelanomaPipeline()
    scores = pipeline.train_all_folds(epochs=5)
    print(f'Mean AUC: {np.mean(scores):.4f}')


def example_custom_model():
    pipeline = MelanomaPipeline()
    scores = pipeline.train_all_folds(model_class=MelanomaClassifierV2, epochs=5)
    print(f'Mean AUC: {np.mean(scores):.4f}')


def example_inference():
    _, test_metadata = load_metadata('data')
    model = load_trained_model('models/melanoma_fold_0.pt', MelanomaClassifier)
    dataset = MelanomaDataset(Path('data/test'), test_metadata, is_training=False)
    predictions = MelanomaInference(model).predict_single_fold(dataset)
    print(f'Generated {len(predictions)} predictions')


def example_ensemble():
    rng = np.random.default_rng(Config.SEED)
    predictions = rng.random((1000, 3))
    targets = rng.binomial(1, 0.1, 1000)
    ensemble = EnsemblePredictor(method='weighted_average')
    # Real ensemble training needs aligned held-out predictions from each model family.
    ensemble.fit(predictions, targets)
    output = ensemble.predict(rng.random((10, 3)))
    print(f'Generated {len(output)} synthetic ensemble predictions')


def example_full_pipeline():
    pipeline = MelanomaPipeline()
    scores, predictions = pipeline.run_full_pipeline(epochs=5, use_tta=True)
    print(f'Mean AUC: {np.mean(scores):.4f}; predictions: {len(predictions)}')


def main():
    examples = {'basic': example_basic_training, 'cv': example_cross_validation,
                'v2': example_custom_model, 'inference': example_inference,
                'ensemble': example_ensemble, 'full': example_full_pipeline}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', choices=examples, nargs='?', default='ensemble')
    args = parser.parse_args()
    examples[args.example]()


if __name__ == '__main__':
    main()
