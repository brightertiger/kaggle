#!/usr/bin/env python3
"""Programmatic examples; running this file shows configuration without training."""
from src.config import Config
from src.data_utils import DataPreprocessor, create_test_loader
from src.models import ModelFactory
from src.pipeline import IMetPipeline
from src.scorer import ModelScorer


def example_basic_usage(data_path='./data', output_path='./output'):
    config = Config(data_path=data_path, output_path=output_path,
                    model_name='resnext101', batch_size=16, epochs=5, num_folds=2)
    return IMetPipeline(config).run_complete_pipeline()


def example_custom_training(data_path='./data', output_path='./output'):
    config = Config(data_path=data_path, output_path=output_path,
                    model_name='resnext50', batch_size=32, epochs=10,
                    learning_rate=5e-4, image_size=256, patience=3, fold_idx=1)
    return IMetPipeline(config).train_model()


def example_data_exploration(config):
    data = DataPreprocessor(config).load_train_data()
    print(data.head())
    print(data['attribute_ids'].str.split().explode().value_counts().head(10))
    return data


def example_model_inference(config, fold=1):
    scorer = ModelScorer(config)
    scorer.load_model(config.get_model_path(fold))
    return scorer.generate_predictions(create_test_loader(config))


def example_ensemble_prediction(config):
    return IMetPipeline(config).generate_predictions()


def example_weighted_ensemble(config, weights):
    # Supply one weight per selected fold; weights are normalized internally.
    return IMetPipeline(config).generate_weighted_predictions(weights)


def example_configuration_examples():
    config = Config(model_name='resnext50', batch_size=32, epochs=15,
                    learning_rate=1e-3, image_size=224, focal_gamma=2.0)
    print(config)
    # Alternative losses are available, but the training recipe uses focal loss.
    return ModelFactory.create_loss_function('f2', config)


if __name__ == '__main__':
    example_configuration_examples()
