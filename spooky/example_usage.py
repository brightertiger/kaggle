#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline import SpookyAuthorPipeline
from src.feature_engineering import TextFeatureEngineer, NaiveBayesFeatureEngineer
from src.models import XGBoostModel, NaiveBayesModel, NeuralNetworkModel
from src.data_utils import DataLoader, DataProcessor
from src.config import Config

def example_text_feature_extraction():
    print("=== Text Feature Extraction Example ===")
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times."
    ]
    
    sample_df = pd.DataFrame({'text': sample_texts})
    
    engineer = TextFeatureEngineer()
    features_df = engineer.extract_all_features(sample_df)
    
    print(f"Extracted {features_df.shape[1]} features from {len(sample_texts)} texts")
    print("\nSample features:")
    print(features_df[['count_word', 'word_length', 'count_punct', 'ratio_stopword']].head())
    
    return features_df

def example_naive_bayes_features():
    print("\n=== Naive Bayes Feature Engineering Example ===")
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times."
    ]
    
    engineer = NaiveBayesFeatureEngineer()
    
    train_texts = sample_texts[:2]
    test_texts = sample_texts[2:]
    
    train_word_feats, test_word_feats = engineer.fit_transform_word_features(train_texts, test_texts)
    train_char_feats, test_char_feats = engineer.fit_transform_char_tfidf_features(train_texts, test_texts)
    train_svd_feats, test_svd_feats = engineer.fit_transform_svd_features(train_texts, test_texts)
    
    print(f"Word features shape: {train_word_feats.shape}")
    print(f"Character TF-IDF features shape: {train_char_feats.shape}")
    print(f"SVD features shape: {train_svd_feats.shape}")
    
    return train_word_feats, test_word_feats

def example_model_training():
    print("\n=== Model Training Example ===")
    
    np.random.seed(42)
    n_samples = 100
    
    X_train = np.random.randn(n_samples, 20)
    y_train = np.random.randint(0, 3, n_samples)
    X_test = np.random.randn(50, 20)
    
    print("Training Naive Bayes model...")
    nb_model = NaiveBayesModel()
    train_score, test_score = nb_model.train_cv(X_train, y_train, X_test)
    
    print(f"Naive Bayes predictions shape: {train_score.shape}")
    print(f"Sample predictions: {train_score.iloc[:3].values}")
    
    print("\nTraining XGBoost model...")
    train_df = pd.DataFrame(X_train)
    train_df['author'] = y_train
    test_df = pd.DataFrame(X_test)
    
    xgb_model = XGBoostModel()
    xgb_model.train(train_df, target_column='author')
    predictions = xgb_model.predict(test_df)
    
    print(f"XGBoost predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.iloc[:3].values}")
    
    return train_score, predictions

def example_data_processing():
    print("\n=== Data Processing Example ===")
    
    sample_train_data = pd.DataFrame({
        'id': range(5),
        'text': [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "It was the best of times, it was the worst of times.",
            "Call me Ishmael. Some years ago—never mind how long precisely.",
            "It is a truth universally acknowledged."
        ],
        'author': ['EAP', 'HPL', 'MWS', 'EAP', 'HPL']
    })
    
    sample_test_data = pd.DataFrame({
        'id': range(5, 8),
        'text': [
            "The sun was shining on the sea.",
            "Once upon a time in a galaxy far, far away.",
            "To be or not to be, that is the question."
        ]
    })
    
    data_loader = DataLoader()
    train_encoded = data_loader.encode_authors(sample_train_data)
    
    print("Original authors:", sample_train_data['author'].tolist())
    print("Encoded authors:", train_encoded['author'].tolist())
    
    return train_encoded, sample_test_data

def example_pipeline_step_by_step():
    print("\n=== Step-by-Step Pipeline Example ===")
    
    pipeline = SpookyAuthorPipeline(data_dir=Path('data'), model_dir=Path('models'), score_dir=Path('scores'))
    
    try:
        print("Step 1: Text Feature Extraction")
        train_features, test_features = pipeline.extract_text_features()
        print(f"✅ Extracted {train_features.shape[1]} text features")
        
        print("\nStep 2: Naive Bayes Training")
        train_nb_score, test_nb_score = pipeline.train_naive_bayes_models()
        print(f"✅ Trained Naive Bayes models")
        
        print("\nStep 3: Neural Network Training")
        train_nn_score, test_nn_score = pipeline.train_neural_network_models()
        print(f"✅ Trained Neural Network model")
        
        print("\nStep 4: LSTM Training")
        train_lstm_score, test_lstm_score = pipeline.train_lstm_model()
        print(f"✅ Trained LSTM model")
        
        print("\nStep 5: XGBoost Training")
        train_data, predictions = pipeline.train_xgboost_model()
        print(f"✅ Trained XGBoost model")
        
        print(f"\n🎉 All steps completed successfully!")
        print(f"Final predictions shape: {predictions.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please ensure data directory contains:")
        print("- data/train.csv")
        print("- data/test.csv")
        print("- glove/glove.6B.50d.txt (for neural networks)")

def example_full_pipeline():
    print("\n=== Full Pipeline Example ===")
    
    pipeline = SpookyAuthorPipeline(data_dir=Path('data'), model_dir=Path('models'), score_dir=Path('scores'))
    
    try:
        fold_scores, predictions = pipeline.run_full_pipeline()
        
        print(f"🎉 Full pipeline completed successfully!")
        print(f"📊 Cross-validation scores: {fold_scores}")
        print(f"📁 Generated {len(predictions)} predictions")
        
        print("\nSample predictions:")
        print(predictions.head())
        
        print("\nPrediction distribution:")
        print(predictions[Config.AUTHOR_NAMES].describe())
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please ensure you have the required data structure.")
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")

def example_feature_importance():
    print("\n=== Feature Importance Example ===")
    
    pipeline = SpookyAuthorPipeline(data_dir=Path('data'), model_dir=Path('models'), score_dir=Path('scores'))
    
    try:
        importance = pipeline.get_feature_importance()
        
        print("Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(importance[:10]):
            print(f"{i+1:2d}. {feature}: {score:.2f}")
        
        print("\nTop 10 Least Important Features:")
        for i, (feature, score) in enumerate(importance[-10:]):
            print(f"{i+1:2d}. {feature}: {score:.2f}")
        
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error getting feature importance: {str(e)}")

def main():
    print("Spooky Author Identification - Example Usage")
    print("=" * 60)
    
    try:
        example_text_feature_extraction()
        example_naive_bayes_features()
        example_model_training()
        example_data_processing()
        example_pipeline_step_by_step()
        example_full_pipeline()
        example_feature_importance()
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("This is expected if data files are not available.")
        print("Please ensure you have the required data structure.")

if __name__ == "__main__":
    main()
