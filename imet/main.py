#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.pipeline import IMetPipeline


def main():
    parser = argparse.ArgumentParser(description='iMet Collection 2019 - FGVC6 Competition Solution')
    
    parser.add_argument('--step', type=str, required=True,
                       choices=['preprocess', 'train', 'predict', 'all'],
                       help='Pipeline step to execute')
    
    parser.add_argument('--model', type=str, default='resnext101',
                       choices=['resnext50', 'resnext101'],
                       help='Model architecture to use')
    
    parser.add_argument('--data-path', type=str, default='../data',
                       help='Path to data directory')
    
    parser.add_argument('--output-path', type=str, default='../output',
                       help='Path to output directory')
    
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    parser.add_argument('--image-size', type=int, default=300,
                       help='Image size for training')
    
    parser.add_argument('--folds', type=int, default=10,
                       help='Number of cross-validation folds')
    
    parser.add_argument('--fold-idx', type=int, default=None,
                       help='Specific fold to train (if None, trains all folds)')
    
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone during initial training')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda:0, etc.)')
    
    parser.add_argument('--num-workers', type=int, default=6,
                       help='Number of data loading workers')
    
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    parser.add_argument('--seed', type=int, default=2017,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    config = Config()
    config.update_from_args(args)
    
    pipeline = IMetPipeline(config)
    
    try:
        if args.step == 'preprocess':
            print("🔄 Starting data preprocessing...")
            pipeline.preprocess_data()
            print("✅ Data preprocessing completed!")
            
        elif args.step == 'train':
            print(f"🚀 Starting model training (model: {args.model})...")
            pipeline.train_model()
            print("✅ Model training completed!")
            
        elif args.step == 'predict':
            print("🔮 Starting prediction generation...")
            pipeline.generate_predictions()
            print("✅ Predictions generated!")
            
        elif args.step == 'all':
            print("🎯 Running complete pipeline...")
            pipeline.preprocess_data()
            pipeline.train_model()
            pipeline.generate_predictions()
            print("🎉 Complete pipeline finished!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
