#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

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
    
    parser.add_argument('--glove_path', type=Path, default=Config.GLOVE_PATH,
                        help='Local GloVe text file matching EMBEDDING_DIM')
    parser.add_argument('--nltk_data_dir', type=Path, default=Config.NLTK_DATA_DIR)
    parser.add_argument('--random_embeddings', action='store_true',
                        help='Use random embeddings instead of loading GloVe')
    parser.add_argument('--nn_epochs', type=int, default=None,
                        help='Override the staged neural training schedule')
    parser.add_argument('--folds', type=int, default=Config.N_FOLDS)
    parser.add_argument('--xgb_rounds', type=int, default=Config.XGB_NUM_ROUNDS)
    args = parser.parse_args()
    if args.folds < 2 or args.xgb_rounds < 1 or (args.nn_epochs is not None and args.nn_epochs < 1):
        parser.error('folds must be at least 2; epochs and rounds must be positive')
    config = Config()
    config.GLOVE_PATH = args.glove_path
    config.NLTK_DATA_DIR = args.nltk_data_dir
    config.RANDOM_EMBEDDINGS = args.random_embeddings
    config.NN_EPOCHS = args.nn_epochs
    config.N_FOLDS = args.folds
    config.XGB_NUM_ROUNDS = args.xgb_rounds
    
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    score_dir = Path(args.score_dir)
    
    pipeline = SpookyAuthorPipeline(data_dir=data_dir, model_dir=model_dir, score_dir=score_dir, config=config)
    
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
            cv_history, predictions = pipeline.run_full_pipeline()
            print(f"🎉 Full pipeline completed successfully!")
            print(f"📁 Models saved to: {model_dir}")
            print(f"📁 Predictions saved to: {score_dir}")
            
        if args.show_importance:
            print('Top feature importances:')
            for feature, score in pipeline.get_feature_importance()[:20]:
                print(f'{feature}: {score:.2f}')

        print(f"\n📊 Final predictions shape: {predictions.shape if 'predictions' in locals() else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
