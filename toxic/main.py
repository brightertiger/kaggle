#!/usr/bin/env python3
"""
Toxic Comment Classification Pipeline
Main entry point for training and inference
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import get_config
from src.pipeline import ToxicCommentPipeline


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Toxic Comment Classification Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'evaluate'], 
                       default='train', help='Pipeline mode')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to custom config file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained models')
    parser.add_argument('--output-path', type=str, default='submissions',
                       help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Override config with command line arguments
    if args.data_path:
        config.data.train_path = os.path.join(args.data_path, 'train.csv')
        config.data.test_path = os.path.join(args.data_path, 'test.csv')
    
    if args.model_path:
        config.model.model_dir = args.model_path
    
    # Create pipeline
    pipeline = ToxicCommentPipeline(config)
    
    if args.mode == 'train':
        print("Starting training pipeline...")
        predictions, results = pipeline.run_full_pipeline(
            train_neural=True, 
            train_traditional=True
        )
        
        print("\nTraining completed!")
        print(f"Best model: {max(results.keys(), key=lambda k: results[k].get('overall', 0))}")
        
    elif args.mode == 'inference':
        print("Inference mode not yet implemented")
        # TODO: Implement inference mode
        
    elif args.mode == 'evaluate':
        print("Evaluation mode not yet implemented")
        # TODO: Implement evaluation mode
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
