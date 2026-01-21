#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import DoodlePipeline
from src.trainer import ModelTrainer
from src.scorer import ModelScorer


def basic_usage_example():
    print("=== Basic Usage Example ===")
    
    config = Config()
    pipeline = DoodlePipeline(config)
    
    train_df = pd.read_csv('../data/train/train.csv')
    valid_df = pd.read_csv('../data/valid/valid.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {valid_df.shape}")
    print(f"Categories: {train_df['word'].nunique()}")


def training_example():
    print("\n=== Training Example ===")
    
    config = Config()
    config.epochs = 5
    config.batch_size = 128
    
    trainer = ModelTrainer(
        config=config,
        model_name='resnet34',
        learning_rate=0.001
    )
    
    train_df = pd.read_csv('../data/train/train.csv', nrows=1000)
    valid_df = pd.read_csv('../data/valid/valid.csv', nrows=200)
    
    results = trainer.train(train_df, valid_df)
    print(f"Training results: {results}")


def prediction_example():
    print("\n=== Prediction Example ===")
    
    config = Config()
    
    test_df = pd.read_csv('../data/test/test_simplified.csv', nrows=100)
    model_path = '../data/model/resnet50/resnet50_best.pth'
    
    scorer = ModelScorer(
        config=config,
        model_path=model_path,
        model_name='resnet50'
    )
    
    submission = scorer.generate_submission(
        test_df, 
        '../data/submit/example_submission.csv'
    )
    print(f"Generated submission with {len(submission)} predictions")


def pipeline_example():
    print("\n=== Full Pipeline Example ===")
    
    config = Config()
    config.epochs = 3
    config.batch_size = 256
    
    pipeline = DoodlePipeline(config)
    
    results = pipeline.run_full_pipeline(
        source_data_path='../data/download',
        test_data_path='../data/test/test_simplified.csv',
        model_name='resnet18',
        learning_rate=0.001
    )
    
    print(f"Pipeline results: {results}")


def custom_model_example():
    print("\n=== Custom Model Configuration ===")
    
    config = Config()
    config.image_size = 128
    config.batch_size = 32
    config.learning_rate = 0.0001
    config.epochs = 100
    config.patience = 10
    
    print("Custom configuration:")
    print(f"  Image size: {config.image_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Patience: {config.patience}")


if __name__ == '__main__':
    basic_usage_example()
    training_example()
    prediction_example()
    pipeline_example()
    custom_model_example()
