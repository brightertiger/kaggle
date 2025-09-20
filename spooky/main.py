#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import SpookyAuthorPipeline
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description='Spooky Author Identification Pipeline')
    
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing training and test data')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--score_dir', type=str, default='scores',
                       help='Directory to save predictions')
    parser.add_argument('--step', type=str, default='full',
                       choices=['text_features', 'naive_bayes', 'neural_network', 'lstm', 'xgboost', 'full'],
                       help='Pipeline step to run')
    parser.add_argument('--show_importance', action='store_true',
                       help='Show feature importance after training')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    score_dir = Path(args.score_dir)
    
    pipeline = SpookyAuthorPipeline(data_dir=data_dir, model_dir=model_dir, score_dir=score_dir)
    
    try:
        if args.step == 'text_features':
            train_features, test_features = pipeline.extract_text_features()
            print(f"✅ Text features extracted: {train_features.shape[1]} features")
            
        elif args.step == 'naive_bayes':
            train_nb_score, test_nb_score = pipeline.train_naive_bayes_models()
            print(f"✅ Naive Bayes models trained")
            
        elif args.step == 'neural_network':
            train_nn_score, test_nn_score = pipeline.train_neural_network_models()
            print(f"✅ Neural Network model trained")
            
        elif args.step == 'lstm':
            train_lstm_score, test_lstm_score = pipeline.train_lstm_model()
            print(f"✅ LSTM model trained")
            
        elif args.step == 'xgboost':
            train_data, predictions = pipeline.train_xgboost_model()
            print(f"✅ XGBoost model trained")
            
        elif args.step == 'full':
            fold_scores, predictions = pipeline.run_full_pipeline()
            print(f"🎉 Full pipeline completed successfully!")
            print(f"📁 Models saved to: {model_dir}")
            print(f"📁 Predictions saved to: {score_dir}")
            
            if args.show_importance:
                print("\n🔍 Top 20 Feature Importances:")
                importance = pipeline.get_feature_importance()
                for i, (feature, score) in enumerate(importance[:20]):
                    print(f"{i+1:2d}. {feature}: {score:.2f}")
        
        print(f"\n📊 Final predictions shape: {predictions.shape if 'predictions' in locals() else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
