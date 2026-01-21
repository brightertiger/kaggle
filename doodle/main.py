#!/usr/bin/env python3

import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import DoodlePipeline


def main():
    parser = argparse.ArgumentParser(description='Quick, Draw! Recognition Pipeline')
    parser.add_argument('--step', 
                       choices=['preprocess', 'train', 'predict', 'all'],
                       default='all',
                       help='Pipeline step to execute')
    parser.add_argument('--model', 
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       default='resnet50',
                       help='Model architecture to use')
    parser.add_argument('--lr', 
                       type=float, 
                       default=0.001,
                       help='Learning rate')
    parser.add_argument('--source-data', 
                       type=str, 
                       default='../data/download',
                       help='Path to source data directory')
    parser.add_argument('--test-data', 
                       type=str, 
                       default='../data/test/test_simplified.csv',
                       help='Path to test data CSV')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=650,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
    pipeline = DoodlePipeline(config)
    
    if args.step in ['preprocess', 'all']:
        print("Preprocessing data...")
        train_df, valid_df = pipeline.preprocess_data(args.source_data)
        print(f"Data preprocessing completed. Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    if args.step in ['train', 'all']:
        print("Training model...")
        train_df = pd.read_csv('../data/train/train.csv')
        valid_df = pd.read_csv('../data/valid/valid.csv')
        
        results = pipeline.train_model(
            train_df, valid_df, args.model, args.lr
        )
        print(f"Training completed. Best accuracy: {results['best_metric']:.4f}")
    
    if args.step in ['predict', 'all']:
        print("Generating predictions...")
        test_df = pd.read_csv(args.test_data)
        model_path = f'../data/model/{args.model}/{args.model}_best.pth'
        output_path = f'../data/submit/{args.model}_submission.csv'
        
        submission = pipeline.generate_predictions(
            test_df, model_path, args.model, output_path
        )
        print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    main()
