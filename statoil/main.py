"""CLI for the original radar CNN / VGG16 / XGBoost pipeline."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Contains download/train.json and download/test.json')
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    parser.add_argument('--submission-dir', type=Path, default=Path('submissions'))
    parser.add_argument('--stage', choices=['all', 'prepare', 'features', 'train', 'predict', 'ensemble', 'xgboost'], default='all')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, help='Override epochs in both VGG stages and both CNNs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--steps-per-epoch', type=int)
    parser.add_argument('--feature-workers', type=int, default=2)
    parser.add_argument('--no-pretrained', action='store_true', help='Initialize VGG16 without downloading ImageNet weights')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU devices')
    args = parser.parse_args()
    if args.folds < 2 or args.batch_size < 1 or args.feature_workers < 1:
        parser.error('folds must be >= 2; batch-size and feature-workers must be positive')
    if any(value is not None and value < 1 for value in (args.epochs, args.steps_per_epoch)):
        parser.error('epochs and steps-per-epoch must be positive')
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
    from src.config import Config
    from src.pipeline import IcebergPipeline
    if args.cpu:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    config = Config()
    config.DATA_DIR = str(args.data_dir)
    config.MODEL_DIR = str(args.model_dir)
    config.SUBMISSION_DIR = str(args.submission_dir)
    config.FOLDS = args.folds
    config.BATCH_SIZE = args.batch_size
    config.FEATURE_WORKERS = args.feature_workers
    if args.no_pretrained:
        config.VGG_WEIGHTS = None
    for settings in config.MODEL_CONFIGS.values():
        if args.epochs is not None:
            settings['epochs'] = args.epochs
            if 'fine_tune_epochs' in settings:
                settings['fine_tune_epochs'] = args.epochs
        if args.steps_per_epoch is not None:
            settings['steps_per_epoch'] = args.steps_per_epoch
    pipeline = IcebergPipeline(config)
    stages = {'all': pipeline.run_full_pipeline, 'prepare': pipeline.prepare_data,
              'features': pipeline.create_xgboost_features, 'train': pipeline.train_models,
              'predict': pipeline.generate_predictions, 'ensemble': pipeline.create_ensemble,
              'xgboost': pipeline.create_xgboost_submission}
    stages[args.stage]()


if __name__ == '__main__':
    main()
