#!/usr/bin/env python3
"""Command line entry point for preprocessing, staged training and inference."""
import argparse

from src.config import Config
from src.pipeline import APTOSPipeline


def main(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description='APTOS Diabetic Retinopathy Detection Pipeline')
    parser.add_argument('--data-dir', default=defaults.DATA_ROOT, help='Parent of pretrain/ and train/')
    parser.add_argument('--pretrain-dir', help='Override the external 2015 data directory')
    parser.add_argument('--train-dir', help='Override the directory containing APTOS train.csv and images')
    parser.add_argument('--model-dir', default=defaults.MODEL_SAVE_PATH)
    parser.add_argument('--image-size', type=int, default=defaults.IMAGE_SIZE)
    parser.add_argument('--large-image-size', type=int, default=defaults.LARGE_IMAGE_SIZE)
    parser.add_argument('--batch-size', type=int, default=defaults.BATCH_SIZE)
    parser.add_argument('--validation-batch-size', type=int, default=defaults.VALIDATION_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=defaults.LEARNING_RATE)
    parser.add_argument('--epochs-pretrain', type=int, default=defaults.NUM_EPOCHS_PRETRAIN)
    parser.add_argument('--epochs-train', type=int, default=defaults.NUM_EPOCHS_TRAIN)
    parser.add_argument('--epochs-combine', type=int, default=defaults.NUM_EPOCHS_COMBINE)
    parser.add_argument('--pretrain-folds', type=int, default=defaults.PRETRAIN_FOLDS)
    parser.add_argument('--train-folds', type=int, default=defaults.TRAIN_FOLDS)
    parser.add_argument('--model-name', default=defaults.MODEL_NAME)
    parser.add_argument('--no-pretrained', action='store_true', help='Random initialization; no weight download')
    parser.add_argument('--device', default=defaults.DEVICE, help='cpu or cuda device; unavailable CUDA falls back to CPU')
    parser.add_argument('--num-workers', type=int, default=defaults.NUM_WORKERS)
    parser.add_argument('--no-apex', action='store_true', help='Use full precision even if NVIDIA Apex is installed')
    parser.add_argument('--noise-augmentation', action='store_true', help='Use paired-view consistency in combined training')
    parser.add_argument('--step', choices=['preprocess', 'pretrain', 'train', 'combine', 'predict', 'all'], default='all')
    parser.add_argument('--fold', type=int, help='One-based fold; omitted means all folds')
    parser.add_argument('--checkpoints', nargs='+', help='Checkpoints to average for prediction')
    parser.add_argument('--test-csv', help='Defaults to test.csv in the APTOS directory')
    parser.add_argument('--test-images', help='Defaults to test_images/ in the APTOS directory')
    parser.add_argument('--submission', help='Defaults to submission.csv in the model directory')
    args = parser.parse_args(argv)
    if args.step == 'predict' and not args.checkpoints:
        parser.error('--step predict requires --checkpoints')
    if args.fold is not None:
        limit = args.pretrain_folds if args.step == 'pretrain' else args.train_folds
        if args.step == 'all':
            limit = min(args.pretrain_folds, args.train_folds)
        if not 1 <= args.fold <= limit:
            parser.error(f'--fold must be between 1 and {limit}')
    config = Config(
        DATA_ROOT=args.data_dir, PRETRAIN_DATA_PATH=args.pretrain_dir,
        TRAIN_DATA_PATH=args.train_dir, MODEL_SAVE_PATH=args.model_dir,
        IMAGE_SIZE=args.image_size, LARGE_IMAGE_SIZE=args.large_image_size,
        BATCH_SIZE=args.batch_size, VALIDATION_BATCH_SIZE=args.validation_batch_size,
        LEARNING_RATE=args.learning_rate, NUM_EPOCHS_PRETRAIN=args.epochs_pretrain,
        NUM_EPOCHS_TRAIN=args.epochs_train, NUM_EPOCHS_COMBINE=args.epochs_combine,
        PRETRAIN_FOLDS=args.pretrain_folds, TRAIN_FOLDS=args.train_folds,
        MODEL_NAME=args.model_name, PRETRAINED=not args.no_pretrained,
        DEVICE=args.device, NUM_WORKERS=args.num_workers, USE_APEX=not args.no_apex,
        USE_NOISE_AUGMENTATION=args.noise_augmentation,
    )
    pipeline = APTOSPipeline(config)
    if args.step == 'preprocess':
        pipeline.preprocess_data()
    elif args.step == 'all':
        pipeline.run_full_pipeline(args.fold)
    elif args.step == 'predict':
        pipeline.predict(args.checkpoints, args.test_csv, args.test_images, args.submission)
    else:
        single = {'pretrain': pipeline._pretrain_fold, 'train': pipeline._train_fold,
                  'combine': pipeline._combine_fold}
        multiple = {'pretrain': pipeline.pretrain_models, 'train': pipeline.train_models,
                    'combine': pipeline.combine_training}
        if args.fold is not None:
            single[args.step](args.fold)
        else:
            multiple[args.step]()


if __name__ == '__main__':
    main()
