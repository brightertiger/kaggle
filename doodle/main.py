#!/usr/bin/env python3
"""Command-line entry point for the stroke rasterization / ResNet pipeline."""
import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import Config
from src.pipeline import DoodlePipeline


def main():
    parser = argparse.ArgumentParser(description='Quick, Draw! Recognition Pipeline')
    parser.add_argument('--step', choices=['preprocess', 'train', 'predict', 'all'], default='all')
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet50')
    parser.add_argument('--data-dir', default='data', help='Prepared data and output directory')
    parser.add_argument('--source-data', default='data/train_simplified', help='Directory of category CSVs')
    parser.add_argument('--test-data', default='data/test_simplified.csv')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=650)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0, help='Use 0 for portable CPU loading')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--max-samples-per-class', type=int, help='Read only this many rows per category CSV')
    parser.add_argument('--seed', type=int, default=2017)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1 or args.num_workers < 0:
        parser.error('epochs, batch-size and image-size must be positive; num-workers must be nonnegative')
    if args.max_samples_per_class is not None and args.max_samples_per_class < 1:
        parser.error('max-samples-per-class must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable; use --device cpu')

    config = Config(data_path=args.data_dir)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.image_size = args.image_size
    config.num_workers = args.num_workers
    config.learning_rate = args.lr
    config.pretrained = args.pretrained
    config.train_ratio = args.train_ratio
    config.max_samples_per_class = args.max_samples_per_class
    config.random_seed = args.seed
    if args.device != 'auto':
        config.device = torch.device(args.device)
    pipeline = DoodlePipeline(config)

    if args.step in ['preprocess', 'all']:
        train_df, valid_df = pipeline.preprocess_data(args.source_data)
    if args.step in ['train', 'all']:
        if args.step == 'train':
            train_df = pd.read_csv(Path(config.data_path) / 'train/train.csv', dtype={'key_id': str})
            valid_df = pd.read_csv(Path(config.data_path) / 'valid/valid.csv', dtype={'key_id': str})
        results = pipeline.train_model(train_df, valid_df, args.model)
        print(f"Best validation top-3 accuracy (%): {results['best_metric']:.4f}")
    if args.step in ['predict', 'all']:
        test_df = pd.read_csv(args.test_data, dtype={'key_id': str})
        model_path = Path(config.model_path) / args.model / f'{args.model}_best.pth'
        output_path = Path(config.submit_path) / f'{args.model}_submission.csv'
        pipeline.generate_predictions(test_df, model_path, args.model, output_path)


if __name__ == '__main__':
    main()
