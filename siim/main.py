#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import MelanomaPipeline
from src.models import MelanomaClassifier, MelanomaClassifierV2
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description='SIIM-ISIC Melanoma Classification Pipeline')
    
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing training and test data')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--score_dir', type=str, default='scores',
                       help='Directory to save predictions')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'v2'],
                       help='Model architecture to use')
    parser.add_argument('--use_tta', action='store_true', default=True,
                       help='Use test time augmentation')
    parser.add_argument('--ensemble_method', type=str, default='weighted_average',
                       choices=['weighted_average', 'logistic_regression', 'random_forest'],
                       help='Ensemble method for final predictions')
    
    args = parser.parse_args()
    
    # Select model class
    if args.model_type == 'standard':
        model_class = MelanomaClassifier
    elif args.model_type == 'v2':
        model_class = MelanomaClassifierV2
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Initialize pipeline
    pipeline = MelanomaPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        score_dir=args.score_dir
    )
    
    # Run full pipeline
    try:
        fold_scores, predictions = pipeline.run_full_pipeline(
            model_class=model_class,
            epochs=args.epochs,
            use_tta=args.use_tta
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Cross-validation AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"📁 Models saved to: {args.model_dir}")
        print(f"📁 Predictions saved to: {args.score_dir}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import numpy as np
    main()
