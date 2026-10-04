#!/usr/bin/env python3

import argparse
from pathlib import Path

from src.core import Config
from src.pipeline import SaltSegmentationPipeline

def main():
    """Main entry point for TGS Salt Identification Challenge"""
    
    parser = argparse.ArgumentParser(
        description='TGS Salt Identification Challenge - Deep Learning Pipeline'
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
                       help='Specific fold to train (1..--folds); otherwise train all folds')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--data-dir', type=str, 
                       default=str(Path(__file__).resolve().parent / 'data'),
                       help='Path to data directory')
    
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--image-size', type=int, default=101)
    parser.add_argument('--padded-size', type=int, default=128)
    parser.add_argument('--no-pretrained', action='store_true', help='Random encoder initialization')
    parser.add_argument('--tiny-model', action='store_true', help='Small random U-Net encoder for smoke tests')
    parser.add_argument('--threshold', type=float, default=None, help='Logit cutoff for submit mode')
    args = parser.parse_args()
    if args.fold is not None and args.mode != 'train':
        parser.error('--fold is only supported with --mode train')
    if args.fold is not None and not 1 <= args.fold <= args.folds:
        parser.error('--fold must lie within 1..--folds')
    
    print("TGS Salt Identification Challenge")
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
    config = Config(data_dir=args.data_dir, output_dir=args.output_dir)
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE_TRAIN = args.batch_size
    config.BATCH_SIZE_VALID = args.batch_size
    config.NUM_FOLDS = args.folds
    config.IMAGE_SIZE = args.image_size
    config.PADDED_SIZE = args.padded_size
    config.PRETRAINED = not args.no_pretrained
    config.TINY_MODEL = args.tiny_model
    config.LEARNING_RATE = args.lr
    config.MODEL_NAME = args.model
    
    if args.device != 'auto':
        config.DEVICE = args.device
    
    config.validate()
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
            submission_df = pipeline.create_submission(args.model, args.use_tta, args.threshold)
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
