#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path

from src.config import Config
from src.pipeline import InstacartPipeline


def main():
    parser = argparse.ArgumentParser(description='Instacart Market Basket Analysis Competition Solution')
    
    parser.add_argument('--step', type=str, required=True,
                       choices=['preprocess', 'features', 'targets', 'train-level1', 'train-level2', 
                               'predict', 'full', 'quick'],
                       help='Pipeline step to execute')
    
    parser.add_argument('--data-path', type=str, default='../data',
                       help='Path to data directory')
    
    parser.add_argument('--output-path', type=str, default='../output',
                       help='Path to output directory')
    
    parser.add_argument('--model-path', type=str, default='../models',
                       help='Path to model directory')
    
    parser.add_argument('--random-seed', type=int, default=108,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--model-type', type=str, default='level2',
                       choices=['level1', 'level2'],
                       help='Model type for prediction')
    
    args = parser.parse_args()
    
    config = Config()
    config.update_from_args(args)
    
    pipeline = InstacartPipeline(config)
    
    try:
        if args.step == 'preprocess':
            print("🔄 Starting data preprocessing...")
            pipeline.preprocess_data()
            print("✅ Data preprocessing completed!")
            
        elif args.step == 'features':
            print("🔧 Starting feature engineering...")
            pipeline.create_features()
            print("✅ Feature engineering completed!")
            
        elif args.step == 'targets':
            print("🎯 Creating target variables...")
            pipeline.create_target_variables()
            print("✅ Target variables created!")
            
        elif args.step == 'train-level1':
            print("🚀 Training Level 1 models...")
            pipeline.train_level1_models()
            print("✅ Level 1 models training completed!")
            
        elif args.step == 'train-level2':
            print("🚀 Training Level 2 model...")
            pipeline.train_level2_model()
            print("✅ Level 2 model training completed!")
            
        elif args.step == 'predict':
            print(f"🔮 Generating predictions using {args.model_type} model...")
            pipeline.generate_predictions(args.model_type)
            print("✅ Predictions generated!")
            
        elif args.step == 'full':
            print("🎯 Running complete pipeline...")
            pipeline.run_full_pipeline()
            print("🎉 Complete pipeline finished!")
            
        elif args.step == 'quick':
            print("⚡ Running quick pipeline...")
            pipeline.run_quick_pipeline()
            print("🎉 Quick pipeline finished!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
