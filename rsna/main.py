#!/usr/bin/env python3

import sys
import os
import argparse

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
                       choices=['resnet18', 'resnet50', 'resnet101', 'inception', 'resnext50', 'resnext101', 'efficientnet'],
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
                       default='data',
                       help='Path to data directory')
    
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--no-pretrained', action='store_true', help='Random initialization; no weight download')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--tta', action='store_true', help='Average original and horizontally flipped predictions')
    parser.add_argument('--train-labels', default='stage_1_train.csv')
    parser.add_argument('--sample-submission', default='stage_1_sample_submission.csv')
    parser.add_argument('--train-images', default='train')
    parser.add_argument('--test-images', default='test')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1 or args.folds < 2 or args.workers < 0:
        parser.error('epochs, batch-size and image-size must be positive; folds >= 2; workers >= 0')
    
    print("RSNA Intracranial Hemorrhage Detection")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 50)
    
    config = Config(args.data_dir, args.output_dir)
    config.IMAGE_SIZE = args.image_size
    config.NUM_FOLDS = args.folds
    config.NUM_WORKERS = args.workers
    config.PRETRAINED = not args.no_pretrained
    config.USE_AMP = not args.no_amp
    config.USE_TTA = args.tta
    config.TRAIN_LABELS = args.train_labels
    config.SAMPLE_SUBMISSION = args.sample_submission
    config.TRAIN_IMAGES = args.train_images
    config.TEST_IMAGES = args.test_images
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE_TRAIN = args.batch_size
    config.BATCH_SIZE_VALID = args.batch_size
    config.BATCH_SIZE_INFERENCE = args.batch_size
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
