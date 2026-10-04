"""Programmatic equivalents of the CLI; select one example explicitly."""
import argparse
from src import Config, SaltSegmentationPipeline
from src.data import create_data_loaders
from src.models import create_model
from src.training import ModelTrainer
import torch


def example_basic_usage(config):
    return SaltSegmentationPipeline(config).run_full_pipeline(config.MODEL_NAME, use_tta=True)


def example_single_fold_training(config):
    pipeline = SaltSegmentationPipeline(config)
    pipeline.preprocess_data()
    train, valid = create_data_loaders(1, config)
    return ModelTrainer(1, config).train(config.MODEL_NAME, train, valid)


def example_model_inference(config):
    return SaltSegmentationPipeline(config).run_inference_only(config.MODEL_NAME, use_tta=True)


def example_custom_model(config):
    model = create_model(config.MODEL_NAME, config, pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, config.PADDED_SIZE, config.PADDED_SIZE))
    print(f'{model.__class__.__name__}: output {tuple(output.shape)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', choices=['full', 'fold', 'inference', 'model'])
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--tiny-model', action='store_true')
    args = parser.parse_args()
    config = Config(args.data_dir, args.output_dir)
    config.TINY_MODEL = args.tiny_model
    {'full': example_basic_usage, 'fold': example_single_fold_training,
     'inference': example_model_inference, 'model': example_custom_model}[args.example](config)


if __name__ == '__main__':
    main()
