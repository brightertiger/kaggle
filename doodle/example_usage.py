#!/usr/bin/env python3
"""Programmatic equivalent of main.py; input and output paths are configurable."""
import argparse

from src import Config, DoodlePipeline


def pipeline_example(source_data_path='data/train_simplified',
                     test_data_path='data/test_simplified.csv',
                     data_path='data', pretrained=True):
    config = Config(data_path=data_path)
    config.epochs = 3
    config.batch_size = 32
    config.num_workers = 0
    config.pretrained = pretrained
    return DoodlePipeline(config).run_full_pipeline(
        source_data_path=source_data_path,
        test_data_path=test_data_path,
        model_name='resnet18',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', default='data/train_simplified')
    parser.add_argument('--test-data', default='data/test_simplified.csv')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(pipeline_example(args.source_data, args.test_data, args.data_dir, args.pretrained))
