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
from src.pipeline import SaltSegmentationPipeline

def main():
    """Main entry point for Salt Identification from Aerial Images"""
    
    parser = argparse.ArgumentParser(
        description='Salt Identification from Aerial Images - Deep Learning Pipeline'
    )
    parser.add_argument('--mode', type=str, 
                       choices=['preprocess', 'train', 'predict', 'evaluate', 'submit', 'full'],
                       default='full',
                       help='Pipeline execution mode')
    parser.add_argument('--model', type=str, 
                       choices=['resnet34', 'seresnet34', 'vgg11'],
                       default='seresnet34',
                       help='Model architecture to use')
    parser.add_argument('--device', type=str, 
                       default='auto',
                       help='Device to use (cuda:0, cpu, auto)')
    parser.add_argument('--epochs', type=int, 
                       default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, 
                       default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, 
                       default=1e-3,
                       help='Learning rate')
    parser.add_argument('--fold', type=int, 
                       default=None,
                       help='Specific fold to train (1-5), if not specified all folds are trained')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--data-dir', type=str, 
                       default='../data',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    print("Salt Identification from Aerial Images")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    if args.fold:
        print(f"Fold: {args.fold}")
    print("=" * 50)
    
    # Create configuration
    config = Config()
    config.DATA_DIR = Path(args.data_dir)
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE_TRAIN = args.batch_size
    config.LEARNING_RATE = args.lr
    config.MODEL_NAME = args.model
    
    if args.device != 'auto':
        config.DEVICE = args.device
    
    # Create pipeline
    pipeline = SaltSegmentationPipeline(config)
    
    try:
        if args.mode == 'preprocess':
            pipeline.preprocess_data()
            
        elif args.mode == 'train':
            if args.fold:
                pipeline.train_fold(args.fold, args.model, args.resume)
            else:
                pipeline.train_models(args.model, args.resume)
                
        elif args.mode == 'predict':
            predictions = pipeline.generate_predictions(args.model, args.use_tta)
            print(f"\nPredictions generated for {len(predictions)} folds")
            
        elif args.mode == 'evaluate':
            results = pipeline.evaluate_models(args.model)
            print(f"\nEvaluation completed. Best threshold: {results['best_threshold']:.3f}")
            
        elif args.mode == 'submit':
            submission_df = pipeline.create_submission(args.model, args.use_tta)
            print(f"\nSubmission created with {len(submission_df)} predictions")
            
        elif args.mode == 'full':
            submission_df = pipeline.run_full_pipeline(
                args.model, 
                args.skip_preprocess, 
                args.resume, 
                args.use_tta
            )
            print(f"\nFull pipeline completed. Final submission has {len(submission_df)} predictions")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
