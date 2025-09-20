#!/usr/bin/env python3

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.pipeline import APTOSPipeline
    from src.config import Config
except ImportError as e:
    print(f"Error: Required dependencies not installed. Please install requirements:")
    print(f"pip install -r requirements.txt")
    print(f"Import error: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='APTOS Diabetic Retinopathy Detection Pipeline')
    parser.add_argument('--data-dir', default='../../data', help='Directory containing raw data')
    parser.add_argument('--pretrain-dir', default='../../data/pretrain', help='Directory for pretrain data')
    parser.add_argument('--train-dir', default='../../data/train', help='Directory for train data')
    parser.add_argument('--model-dir', default='../../model', help='Directory for model files')
    parser.add_argument('--image-size', type=int, default=256, help='Image size for training')
    parser.add_argument('--large-image-size', type=int, default=330, help='Large image size for combined training')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs-pretrain', type=int, default=10, help='Number of epochs for pretraining')
    parser.add_argument('--epochs-train', type=int, default=12, help='Number of epochs for training')
    parser.add_argument('--epochs-combine', type=int, default=10, help='Number of epochs for combined training')
    parser.add_argument('--pretrain-folds', type=int, default=10, help='Number of folds for pretrain data')
    parser.add_argument('--train-folds', type=int, default=5, help='Number of folds for train data')
    parser.add_argument('--step', choices=['preprocess', 'pretrain', 'train', 'combine', 'all'], 
                       default='all', help='Pipeline step to run')
    parser.add_argument('--fold', type=int, help='Specific fold to train (if not specified, trains all folds)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.DATA_ROOT = args.data_dir
    config.PRETRAIN_DATA_PATH = args.pretrain_dir
    config.TRAIN_DATA_PATH = args.train_dir
    config.MODEL_SAVE_PATH = args.model_dir
    config.IMAGE_SIZE = args.image_size
    config.LARGE_IMAGE_SIZE = args.large_image_size
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.NUM_EPOCHS_PRETRAIN = args.epochs_pretrain
    config.NUM_EPOCHS_TRAIN = args.epochs_train
    config.NUM_EPOCHS_COMBINE = args.epochs_combine
    config.PRETRAIN_FOLDS = args.pretrain_folds
    config.TRAIN_FOLDS = args.train_folds
    
    # Initialize pipeline
    pipeline = APTOSPipeline(config)
    
    # Run specified step
    if args.step == 'preprocess':
        pipeline.preprocess_data()
    elif args.step == 'pretrain':
        if args.fold:
            pipeline._pretrain_fold(args.fold)
        else:
            pipeline.pretrain_models()
    elif args.step == 'train':
        if args.fold:
            pipeline._train_fold(args.fold)
        else:
            pipeline.train_models()
    elif args.step == 'combine':
        if args.fold:
            pipeline._combine_fold(args.fold)
        else:
            pipeline.combine_training()
    elif args.step == 'all':
        pipeline.run_full_pipeline()

if __name__ == '__main__':
    main()
