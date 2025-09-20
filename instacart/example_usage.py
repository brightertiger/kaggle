#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import Config
from src.pipeline import InstacartPipeline
from src.feature_engineering import UserFeatureEngineer, ProductFeatureEngineer, Word2VecEngineer
from src.models import XGBoostModel, Level2Model, ModelEvaluator


def example_basic_usage():
    print("🎯 Example 1: Basic Pipeline Usage")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../models'
    
    pipeline = InstacartPipeline(config)
    
    if pipeline.validate_setup():
        print("✅ Setup validation passed")
        
        print("🔄 Running preprocessing...")
        pipeline.preprocess_data()
        
        print("🔧 Creating features...")
        pipeline.create_features()
        
        print("🎯 Creating target variables...")
        pipeline.create_target_variables()
        
        print("🚀 Training Word2Vec model...")
        pipeline._train_word2vec_model()
        
        print("🔮 Generating predictions...")
        pipeline._generate_level1_predictions()
        
        print("🎉 Basic pipeline completed!")
    else:
        print("❌ Setup validation failed")


def example_custom_training():
    print("\n🎯 Example 2: Custom Training Configuration")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../models'
    
    config.xgb_params['eta'] = 0.05
    config.xgb_params['max_depth'] = 8
    config.xgb_training_params['num_boost_round'] = 500
    
    print(f"Configuration:\n{config}")
    
    pipeline = InstacartPipeline(config)
    
    if pipeline.validate_setup():
        print("🚀 Training with custom parameters...")
        pipeline._train_word2vec_model()
        print("✅ Custom training completed!")
    else:
        print("❌ Setup validation failed")


def example_feature_engineering():
    print("\n🎯 Example 3: Feature Engineering")
    print("=" * 50)
    
    config = Config()
    
    try:
        orders = pd.read_csv(config.get_data_path('driver', 'driver_order.csv'))
        products = pd.read_csv(config.get_data_path('driver', 'driver_order_products.csv'))
        users = pd.read_csv(config.get_data_path('driver', 'driver_user.csv'))
        
        print("📊 Loading data...")
        print(f"Orders shape: {orders.shape}")
        print(f"Products shape: {products.shape}")
        print(f"Users shape: {users.shape}")
        
        print("\n🔧 Creating user features...")
        user_engineer = UserFeatureEngineer(config)
        
        orders_filtered = orders.drop('eval_set', axis=1)
        products_merged = products.merge(orders_filtered, on='order_id', how='inner')
        
        user_profile = user_engineer.build_user_profile(users, orders, products_merged, None)
        print(f"User profile shape: {user_profile.shape}")
        print(f"User profile columns: {list(user_profile.columns)}")
        
        print("\n🛍️ Creating product features...")
        product_engineer = ProductFeatureEngineer(config)
        product_info = pd.read_csv(config.get_data_path('download', 'products.csv'))
        
        product_profile = product_engineer.build_product_profile(products_merged, product_info)
        print(f"Product profile shape: {product_profile.shape}")
        print(f"Product profile columns: {list(product_profile.columns)[:10]}...")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please run preprocessing first")


def example_model_training():
    print("\n🎯 Example 4: Model Training")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../models'
    
    try:
        print("📊 Loading training data...")
        train_data = pd.read_csv(config.get_data_path('model', 'dependent', 'dependent_n_2.csv'))
        valid_data = pd.read_csv(config.get_data_path('model', 'dependent', 'dependent_n_1.csv'))
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")
        
        print("\n🚀 Training XGBoost model...")
        model = XGBoostModel(config, 'level1')
        
        results = model.train(train_data, valid_data)
        
        print(f"✅ Training completed!")
        print(f"Train AUC: {results['train_auc']:.4f}")
        print(f"Valid AUC: {results['valid_auc']:.4f}")
        
        print(f"\n📊 Top 10 Feature Importance:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.2f}")
        
        model.save_model(config.get_model_path('example_model.model'))
        print(f"💾 Model saved to {config.get_model_path('example_model.model')}")
        
    except FileNotFoundError as e:
        print(f"❌ Training data not found: {e}")
        print("Please run feature engineering first")


def example_model_evaluation():
    print("\n🎯 Example 5: Model Evaluation")
    print("=" * 50)
    
    config = Config()
    
    try:
        print("📊 Loading model and data...")
        model = XGBoostModel(config, 'level1')
        model.load_model(config.get_model_path('example_model.model'))
        
        test_data = pd.read_csv(config.get_data_path('model', 'dependent', 'dependent_n_1.csv'))
        
        print("🔮 Making predictions...")
        predictions = model.predict(test_data)
        
        print("📈 Evaluating model...")
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate_model(test_data['reordered'], predictions)
        
        print("✅ Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        optimal_threshold, best_f1 = evaluator.find_optimal_threshold(test_data['reordered'], predictions)
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Best F1 Score: {best_f1:.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ Model or data not found: {e}")
        print("Please train a model first")


def example_word2vec_features():
    print("\n🎯 Example 6: Word2Vec Feature Engineering")
    print("=" * 50)
    
    config = Config()
    
    try:
        print("📊 Loading product data...")
        product_info = pd.read_csv(config.get_data_path('download', 'products.csv'))
        products = pd.read_csv(config.get_data_path('driver', 'driver_order_products.csv'))
        
        print(f"Product info shape: {product_info.shape}")
        print(f"Products shape: {products.shape}")
        
        print("\n🔧 Creating Word2Vec model...")
        word2vec_engineer = Word2VecEngineer(config)
        
        model, product_embeddings = word2vec_engineer.create_product_embeddings(products, product_info)
        user_embeddings = word2vec_engineer.create_user_embeddings(products, model)
        
        print(f"Product embeddings shape: {product_embeddings.shape}")
        print(f"User embeddings shape: {user_embeddings.shape}")
        
        print("\n📊 Sample embeddings:")
        print(f"Product 1 embedding (first 5 dims): {product_embeddings.iloc[0, :5].values}")
        print(f"User 1 embedding (first 5 dims): {user_embeddings.iloc[0, :5].values}")
        
        print("\n💾 Saving embeddings...")
        product_embeddings.to_csv(config.get_data_path('profile', 'example_prodvecs.csv'), index=False)
        user_embeddings.to_csv(config.get_data_path('profile', 'example_uservecs.csv'), index=False)
        
        print("✅ Word2Vec features created successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please run preprocessing first")


def example_level2_training():
    print("\n🎯 Example 7: Level 2 Model Training")
    print("=" * 50)
    
    config = Config()
    config.data_path = '../data'
    config.output_path = '../output'
    config.model_path = '../models'
    
    try:
        print("📊 Loading Level 2 data...")
        dependent_data = pd.read_csv(config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        
        print(f"Dependent data shape: {dependent_data.shape}")
        
        print("\n🔧 Creating Level 2 features...")
        pipeline = InstacartPipeline(config)
        independent_data = pipeline._create_level2_features()
        
        print(f"Independent data shape: {independent_data.shape}")
        
        print("\n🚀 Training Level 2 model...")
        level2_model = Level2Model(config)
        results = level2_model.train(dependent_data, independent_data)
        
        print(f"✅ Level 2 training completed!")
        print(f"Train AUC: {results['train_auc']:.4f}")
        print(f"Valid AUC: {results['valid_auc']:.4f}")
        
        level2_model.save_model(config.get_model_path('example_level2.model'))
        print(f"💾 Level 2 model saved!")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please run feature engineering first")


def example_submission_generation():
    print("\n🎯 Example 8: Submission Generation")
    print("=" * 50)
    
    config = Config()
    
    try:
        print("📊 Loading model and test data...")
        model = XGBoostModel(config, 'level1')
        model.load_model(config.get_model_path('example_model.model'))
        
        test_data = pd.read_csv(config.get_data_path('model', 'dependent', 'dependent_n_1.csv'))
        
        print("🔮 Making predictions...")
        predictions = model.predict(test_data)
        
        print("📝 Creating submission...")
        from src.models import SubmissionGenerator
        submission_gen = SubmissionGenerator(config)
        
        submission = submission_gen.create_submission(
            predictions, 
            test_data['user_id'].values, 
            test_data['product_id'].values,
            threshold=0.5
        )
        
        print(f"✅ Submission created with {len(submission)} orders")
        print(f"Sample submission:")
        print(submission.head())
        
        submission_path = config.get_output_path('example_submission.csv')
        submission_gen.save_submission(submission, submission_path)
        print(f"💾 Submission saved to {submission_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Model or data not found: {e}")
        print("Please train a model first")


def example_configuration_examples():
    print("\n🎯 Example 9: Configuration Examples")
    print("=" * 50)
    
    print("📋 Default Configuration:")
    default_config = Config()
    print(default_config)
    
    print("\n📋 Custom XGBoost Parameters:")
    custom_config = Config()
    custom_config.xgb_params['eta'] = 0.05
    custom_config.xgb_params['max_depth'] = 10
    custom_config.xgb_params['subsample'] = 0.8
    custom_config.xgb_training_params['num_boost_round'] = 1000
    
    print("XGBoost Parameters:")
    for key, value in custom_config.xgb_params.items():
        print(f"  {key}: {value}")
    
    print("\n📋 Custom Feature Parameters:")
    custom_config.feature_params['word2vec_dim'] = 150
    custom_config.feature_params['word2vec_window'] = 7
    custom_config.encoding_params['prior_probability'] = 0.15
    
    print("Feature Parameters:")
    for key, value in custom_config.feature_params.items():
        print(f"  {key}: {value}")
    
    print("Encoding Parameters:")
    for key, value in custom_config.encoding_params.items():
        print(f"  {key}: {value}")


def main():
    print("🛒 Instacart Market Basket Analysis Examples")
    print("=" * 60)
    
    examples = [
        example_basic_usage,
        example_custom_training,
        example_feature_engineering,
        example_model_training,
        example_model_evaluation,
        example_word2vec_features,
        example_level2_training,
        example_submission_generation,
        example_configuration_examples
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            print("\n" + "=" * 60)
    
    print("\n🎉 All examples completed!")
    print("\n💡 Tips:")
    print("  - Ensure data files are in the correct location")
    print("  - Run preprocessing before feature engineering")
    print("  - Train models before evaluation")
    print("  - Check GPU memory usage during training")
    print("  - Use quick pipeline for testing")


if __name__ == '__main__':
    main()
