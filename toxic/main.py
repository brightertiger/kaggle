#!/usr/bin/env python3
"""Train and predict, preprocess CSVs, or evaluate saved OOF predictions."""

import argparse
from pathlib import Path

from src.config import get_config
from src.data_utils import read_csv
from src.pipeline import ToxicCommentPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'preprocess', 'evaluate'], default='train')
    parser.add_argument('--config', help='JSON file with data/model/evaluation sections')
    parser.add_argument('--data-path', help='Directory containing train.csv and test.csv')
    parser.add_argument('--model-path', help='Directory for out-of-fold prediction CSVs')
    parser.add_argument('--output-path', help='Directory for test submission CSVs')
    parser.add_argument('--models', choices=['all', 'traditional', 'neural'], default='all')
    parser.add_argument('--embeddings', choices=['glove', 'fasttext'], help='Pretrained embedding source')
    parser.add_argument('--embedding-path', help='Local text vectors for the selected source')
    parser.add_argument('--random-embeddings', action='store_true', help='Use random vectors for smoke tests')
    parser.add_argument('--cpu', action='store_true', help='Disable TensorFlow GPU devices')
    args = parser.parse_args()

    config = get_config(args.config)
    if args.data_path:
        config.data.train_path = str(Path(args.data_path, 'train.csv'))
        config.data.test_path = str(Path(args.data_path, 'test.csv'))
    if args.model_path:
        config.model.model_dir = args.model_path
    if args.output_path:
        config.model.submission_dir = args.output_path
    if args.embeddings:
        config.model.embedding_type = args.embeddings
    if args.embedding_path:
        setattr(config.model, f'{config.model.embedding_type}_path', args.embedding_path)
    if args.random_embeddings:
        config.model.random_embeddings = True
    if args.cpu:
        config.model.cpu_only = True

    pipeline = ToxicCommentPipeline(config)
    if args.mode == 'preprocess':
        pipeline.prepare_data()
    elif args.mode == 'evaluate':
        paths = sorted(Path(config.model.model_dir).glob('*_validation.csv'))
        if not paths:
            parser.error(f'No *_validation.csv files in {config.model.model_dir}; run training first')
        predictions = {path.stem.removesuffix('_validation'): read_csv(path) for path in paths}
        results = pipeline.evaluate_models(predictions)
        pipeline.evaluator.print_evaluation_results(results)
    else:
        _, results = pipeline.run_full_pipeline(
            train_neural=args.models in {'all', 'neural'},
            train_traditional=args.models in {'all', 'traditional'},
        )
        best, score = pipeline.evaluator.find_best_model(results)
        print(f'Best OOF model: {best}, mean ROC AUC: {score:.4f}')


if __name__ == '__main__':
    main()
