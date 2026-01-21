import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import run_full_pipeline, create_folds, resample_audio, ensemble_predictions

def main():
    parser = argparse.ArgumentParser(description='RFCX Species Audio Detection Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--model', type=str, choices=['resnet', 'resnest'], 
                       default='resnet', help='Model architecture')
    parser.add_argument('--tta', action='store_true', help='Apply test time augmentation')
    parser.add_argument('--ensemble', action='store_true', help='Create ensemble predictions')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    config = Config()
    config.training.num_folds = args.folds
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    
    if args.mode == 'train':
        from src.trainer import train_model
        train_model(config, args.model)
    elif args.mode == 'predict':
        from src.predictor import generate_predictions
        generate_predictions(config, args.model, args.tta)
    else:
        run_full_pipeline(config, args.model, args.tta, args.ensemble)

if __name__ == "__main__":
    main()
