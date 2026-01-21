#!/usr/bin/env python3

"""
TalkingData AdTracking Fraud Detection

Main entry point for the TalkingData AdTracking Fraud Detection pipeline.
This script provides a command-line interface for running the complete machine learning
pipeline for detecting fraudulent ad clicks in mobile advertising.

Usage:
    python main.py --mode full --data-dir ../data
    python main.py --mode preprocess --data-dir ../data
    python main.py --mode train --data-dir ../data
    python main.py --mode predict --data-dir ../data
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from src.core import Config
from src.pipeline import TalkingDataPipeline

def main():
    """Main entry point for TalkingData AdTracking Fraud Detection pipeline."""
    
    parser = argparse.ArgumentParser(
        description='TalkingData AdTracking Fraud Detection - Machine Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --mode full --data-dir ../data
  
  # Preprocess data only
  python main.py --mode preprocess --data-dir ../data
  
  # Train models only
  python main.py --mode train --data-dir ../data
  
  # Generate predictions only
  python main.py --mode predict --data-dir ../data
  
  # Evaluate models
  python main.py --mode evaluate --data-dir ../data
        """
    )
    
    parser.add_argument('--mode', type=str, 
                       choices=['preprocess', 'train', 'predict', 'evaluate', 'submit', 'full'],
                       default='full',
                       help='Pipeline execution mode (default: full)')
    
    parser.add_argument('--data-dir', type=str, 
                       default='../data',
                       help='Path to data directory (default: ../data)')
    
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip data preprocessing step (for full mode)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print header
    print("TalkingData AdTracking Fraud Detection")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Data Directory: {args.data_dir}")
    if args.skip_preprocess:
        print("Skipping preprocessing step")
    print("=" * 50)
    
    try:
        # Create configuration
        config = Config()
        config.DATA_DIR = Path(args.data_dir)
        
        # Create pipeline
        pipeline = TalkingDataPipeline(config)
        
        # Execute based on mode
        if args.mode == 'preprocess':
            print("\nRunning data preprocessing...")
            pipeline.preprocess_data()
            print("Data preprocessing completed successfully!")
            
        elif args.mode == 'train':
            print("\nTraining models...")
            # Check if feature-engineered data exists
            if not (config.MODELS_DIR / 'train_data.feather').exists():
                print("Feature-engineered data not found. Running preprocessing first...")
                pipeline.create_feature_engineered_data()
            
            # Load data and train
            train_data = pd.read_feather(config.MODELS_DIR / 'train_data.feather')
            valid_data = pd.read_feather(config.MODELS_DIR / 'valid_data.feather')
            pipeline.trainer.train_multiple_models(train_data, valid_data, ['model_1', 'model_2'])
            print("Model training completed successfully!")
            
        elif args.mode == 'predict':
            print("\nGenerating predictions...")
            pipeline.generate_predictions()
            print("Prediction generation completed successfully!")
            
        elif args.mode == 'evaluate':
            print("\nEvaluating models...")
            evaluations = pipeline.evaluate_models()
            
            print("\nModel Evaluation Results:")
            print("-" * 30)
            for model_name, eval_results in evaluations.items():
                print(f"{model_name}:")
                print(f"  AUC Score: {eval_results['auc_score']:.4f}")
                print(f"  Best Threshold: {eval_results['best_threshold']:.4f}")
            
            # Show feature importance
            print("\nTop 10 Most Important Features:")
            print("-" * 35)
            importance = pipeline.get_feature_importance()
            for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
                
        elif args.mode == 'submit':
            print("\nCreating ensemble submission...")
            submission = pipeline.create_ensemble_submission()
            print(f"Ensemble submission created with {len(submission)} predictions")
            
        elif args.mode == 'full':
            print("\nRunning complete pipeline...")
            submission = pipeline.run_full_pipeline(skip_preprocess=args.skip_preprocess)
            print(f"\nPipeline completed successfully!")
            print(f"Final submission saved with {len(submission)} predictions")
            
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Please ensure all data files are available in the specified directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
