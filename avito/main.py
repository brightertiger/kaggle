#!/usr/bin/env python3

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.pipeline import AvitoPipeline
    from src.config import Config
except ImportError as e:
    print(f"Error: Required dependencies not installed. Please install requirements:")
    print(f"pip install -r requirements.txt")
    print(f"Import error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Avito Deal Probability Prediction Pipeline')
    parser.add_argument('--data-dir', default='../../data', help='Directory containing raw data')
    parser.add_argument('--features-dir', default='../../data/data/features', help='Directory for feature files')
    parser.add_argument('--model-dir', default='../../model', help='Directory for model files')
    parser.add_argument('--output-dir', default='../../output', help='Directory for output files')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--random-state', type=int, default=2017, help='Random state for reproducibility')
    parser.add_argument('--step', choices=['preprocess', 'features', 'train', 'evaluate', 'submission', 'all'], 
                       default='all', help='Pipeline step to run')
    parser.add_argument('--fold', type=int, help='Specific fold to train (if not specified, trains all folds)')
    
    args = parser.parse_args()
    
    config = Config()
    config.avito.DATA_ROOT = args.data_dir
    config.avito.FEATURES_DIR = args.features_dir
    config.avito.MODEL_DIR = args.model_dir
    config.avito.OUTPUT_DIR = args.output_dir
    config.avito.N_FOLDS = args.n_folds
    config.avito.RANDOM_STATE = args.random_state
    
    pipeline = AvitoPipeline(config)
    
    if args.step == 'preprocess':
        pipeline.preprocess_data()
    elif args.step == 'features':
        pipeline.generate_features()
    elif args.step == 'train':
        pipeline.train_models()
    elif args.step == 'evaluate':
        pipeline.evaluate_pipeline()
    elif args.step == 'submission':
        pipeline.generate_submission()
    elif args.step == 'all':
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
