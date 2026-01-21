#!/usr/bin/env python3

import sys
import os
import torch
import pandas as pd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from src.core import Config
from src.pipeline import IntracranialHemorrhagePipeline

def main():
    """Main entry point for RSNA Intracranial Hemorrhage Detection"""
    
    parser = argparse.ArgumentParser(
        description='RSNA Intracranial Hemorrhage Detection - Deep Learning Pipeline'
    )
    parser.add_argument('--mode', type=str, 
                       choices=['preprocess', 'train', 'predict', 'full', 'analyze', 'validate'],
                       default='full',
                       help='Pipeline execution mode')
    parser.add_argument('--model', type=str, 
                       choices=['resnet50', 'resnet101', 'inception', 'resnext50', 'resnext101', 'efficientnet'],
                       default='resnext101',
                       help='Model architecture to use')
    parser.add_argument('--device', type=str, 
                       default='auto',
                       help='Device to use (cuda:0, cpu, auto)')
    parser.add_argument('--epochs', type=int, 
                       default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, 
                       default=12,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, 
                       default=1e-4,
                       help='Learning rate')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--data-dir', type=str, 
                       default='../data',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    print("RSNA Intracranial Hemorrhage Detection")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 50)
    
    config = Config()
    config.DATA_DIR = Path(args.data_dir)
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE_TRAIN = args.batch_size
    config.LEARNING_RATE = args.lr
    
    if args.device != 'auto':
        config.DEVICE = args.device
    
    pipeline = IntracranialHemorrhagePipeline(config)
    
    try:
        if args.mode == 'preprocess':
            pipeline.preprocess_data()
        elif args.mode == 'train':
            pipeline.train_models(args.model)
        elif args.mode == 'predict':
            submission_df = pipeline.generate_predictions(args.model)
            print(f"\nSubmission file created with {len(submission_df)} predictions")
        elif args.mode == 'full':
            submission_df = pipeline.run_full_pipeline(args.model, args.skip_preprocess)
            print(f"\nFinal submission created with {len(submission_df)} predictions")
        elif args.mode == 'analyze':
            from src.data import analyze_dataset
            analyze_dataset(config)
        elif args.mode == 'validate':
            from src.inference import run_validation
            run_validation(config, args.model)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
