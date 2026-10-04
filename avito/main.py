#!/usr/bin/env python3
"""Command-line entry point for the compact Avito pipeline."""
import argparse

from src.config import AvitoConfig, Config
from src.pipeline import AvitoPipeline


def main():
    defaults = AvitoConfig()
    parser = argparse.ArgumentParser(description='Avito Deal Probability Prediction Pipeline')
    parser.add_argument('--data-dir', default=defaults.DATA_ROOT,
                        help='Directory containing train/test and active CSV files')
    parser.add_argument('--output-dir', default=defaults.OUTPUT_DIR,
                        help='Submission and default artifact directory')
    parser.add_argument('--features-dir', help='Feature directory (default: OUTPUT_DIR/features)')
    parser.add_argument('--model-dir', help='Folds and prediction directory (default: OUTPUT_DIR/models)')
    parser.add_argument('--n-folds', type=int, default=defaults.N_FOLDS)
    parser.add_argument('--random-state', type=int, default=defaults.RANDOM_STATE)
    parser.add_argument('--step', choices=['preprocess', 'features', 'train', 'evaluate', 'submission', 'all'],
                        default='all')
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error('--n-folds must be at least 2')
    from pathlib import Path
    output = Path(args.output_dir)
    config = Config(AvitoConfig(
        DATA_ROOT=args.data_dir, OUTPUT_DIR=str(output),
        FEATURES_DIR=args.features_dir or str(output / 'features'),
        MODEL_DIR=args.model_dir or str(output / 'models'),
        N_FOLDS=args.n_folds, RANDOM_STATE=args.random_state))
    pipeline = AvitoPipeline(config)
    actions = {'preprocess': pipeline.preprocess_data, 'features': pipeline.generate_features,
               'train': pipeline.train_models, 'evaluate': pipeline.evaluate_pipeline,
               'submission': pipeline.generate_submission, 'all': pipeline.run_full_pipeline}
    result = actions[args.step]()
    if args.step == 'evaluate':
        print(result)


if __name__ == '__main__':
    main()
