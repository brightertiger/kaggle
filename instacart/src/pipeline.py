#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .config import Config
from .data_utils import DataLoader, TargetCreator, create_train_valid_split
from .feature_engineering import (
    UserFeatureEngineer, ProductFeatureEngineer, MeanEncodingEngineer, 
    InteractionFeatureEngineer, Word2VecEngineer
)
from .models import XGBoostModel, Level1Ensemble, Level2Model, ModelEvaluator, SubmissionGenerator


class InstacartPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.target_creator = TargetCreator(config)
        
        self.user_engineer = UserFeatureEngineer(config)
        self.product_engineer = ProductFeatureEngineer(config)
        self.encoding_engineer = MeanEncodingEngineer(config)
        self.interaction_engineer = InteractionFeatureEngineer(config)
        self.word2vec_engineer = Word2VecEngineer(config)
        
        self.evaluator = ModelEvaluator(config)
        self.submission_generator = SubmissionGenerator(config)
        
        self.models = {}
        self.features = {}
    
    def validate_setup(self) -> bool:
        try:
            orders_path = self.config.get_data_path('download', 'orders.csv')
            products_path = self.config.get_data_path('download', 'products.csv')
            
            if not os.path.exists(orders_path):
                print(f"❌ Orders file not found: {orders_path}")
                return False
            
            if not os.path.exists(products_path):
                print(f"❌ Products file not found: {products_path}")
                return False
            
            print("✅ Data files validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Setup validation failed: {e}")
            return False
    
    def preprocess_data(self):
        print("🔄 Starting data preprocessing...")
        
        orders, products, order_products_prior, order_products_train = self.data_loader.load_raw_data()
        
        print("📊 Creating user splits...")
        users = self.data_loader.create_user_splits(orders)
        
        print("📊 Creating order features...")
        orders_processed = self.data_loader.create_order_features(orders)
        
        print("📊 Creating product features...")
        products_processed = self.data_loader.create_product_features(products)
        
        print("📊 Creating order-products data...")
        order_products = self.data_loader.create_order_products(
            order_products_prior, order_products_train, orders_processed
        )
        
        print("💾 Saving processed data...")
        self.data_loader.save_processed_data(users, orders_processed, products_processed, order_products)
        
        print("✅ Data preprocessing completed!")
    
    def create_features(self):
        print("🔧 Starting feature engineering...")
        
        users = pd.read_csv(self.config.get_data_path('driver', 'driver_user.csv'))
        orders = pd.read_csv(self.config.get_data_path('driver', 'driver_order.csv'))
        products = pd.read_csv(self.config.get_data_path('driver', 'driver_order_products.csv'))
        product_info = pd.read_csv(self.config.get_data_path('download', 'products.csv'))
        
        orders_with_features = orders.drop('eval_set', axis=1)
        products_merged = products.merge(orders_with_features, on='order_id', how='inner')
        products_merged = products_merged.merge(users, on='user_id', how='inner')
        
        print("👤 Creating user features...")
        user_profile = self.user_engineer.build_user_profile(users, orders, products_merged, None)
        user_profile.to_csv(self.config.get_data_path('profile', 'user_profile.csv'), index=False)
        
        print("🛍️ Creating product features...")
        product_profile = self.product_engineer.build_product_profile(products_merged, product_info)
        product_profile.to_csv(self.config.get_data_path('profile', 'product_basic_profile.csv'), index=False)
        
        print("📈 Creating mean encoding features...")
        product_encoding, user_encoding = self.encoding_engineer.create_encoding_features(products_merged)
        product_encoding.to_csv(self.config.get_data_path('profile', 'product_brrc_profile.csv'), index=False)
        user_encoding.to_csv(self.config.get_data_path('profile', 'user_brrc_profile.csv'), index=False)
        
        print("🔄 Creating interaction features...")
        user_product_profile = self._create_user_product_profile(products_merged)
        interaction_features = self.interaction_engineer.create_user_product_features(
            None, orders, user_product_profile
        )
        interaction_features.to_csv(self.config.get_data_path('profile', 'user_product_2way.csv'), index=False)
        
        print("📝 Creating Word2Vec embeddings...")
        word2vec_model, product_embeddings = self.word2vec_engineer.create_product_embeddings(
            products_merged, product_info
        )
        user_embeddings = self.word2vec_engineer.create_user_embeddings(products_merged, word2vec_model)
        
        product_embeddings.to_csv(self.config.get_data_path('profile', 'prodvecs.csv'), index=False)
        user_embeddings.to_csv(self.config.get_data_path('profile', 'uservecs.csv'), index=False)
        
        word2vec_model.save(self.config.get_data_path('profile', 'word2vec_model.model'))
        
        print("✅ Feature engineering completed!")
    
    def _create_user_product_profile(self, products: pd.DataFrame) -> pd.DataFrame:
        user_product_profile = products.groupby(['user_id', 'product_id']).agg({
            'order_id': 'count',
            'reordered': 'mean'
        }).reset_index()
        
        user_product_profile.columns = ['user_id', 'product_id', 'usr_prd_cnt', 'usr_prd_reorder_rate']
        
        return user_product_profile
    
    def create_target_variables(self):
        print("🎯 Creating target variables...")
        
        products = pd.read_csv(self.config.get_data_path('driver', 'driver_order_products.csv'))
        orders = pd.read_csv(self.config.get_data_path('driver', 'driver_order.csv'))
        
        self.target_creator.create_dependent_datasets(products, orders)
        
        print("✅ Target variables created!")
    
    def train_level1_models(self):
        print("🚀 Training Level 1 models...")
        
        self._train_word2vec_model()
        self._train_basic_features_model()
        self._train_ensemble_model()
        
        print("✅ Level 1 models training completed!")
    
    def _train_word2vec_model(self):
        print("📝 Training Word2Vec model...")
        
        prod_vecs = pd.read_csv(self.config.get_data_path('profile', 'prodvecs.csv'))
        user_vecs = pd.read_csv(self.config.get_data_path('profile', 'uservecs.csv'))
        
        train_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n_2.csv'))
        valid_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n_1.csv'))
        
        train_data = train_data.merge(prod_vecs, on='product_id', how='inner')
        train_data = train_data.merge(user_vecs, on='user_id', how='inner')
        valid_data = valid_data.merge(prod_vecs, on='product_id', how='inner')
        valid_data = valid_data.merge(user_vecs, on='user_id', how='inner')
        
        model = XGBoostModel(self.config, 'level1')
        results = model.train(train_data, valid_data)
        
        model.save_model(self.config.get_model_path('word2vec_xgb.model'))
        
        print(f"Word2Vec Model - Train AUC: {results['train_auc']:.4f}, Valid AUC: {results['valid_auc']:.4f}")
    
    def _train_basic_features_model(self):
        print("🔧 Training basic features model...")
        
        dependent_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        independent_data = self._create_independent_features()
        
        train_data = dependent_data[dependent_data['eval_set'] == 'train']
        valid_data = dependent_data[dependent_data['eval_set'] == 'valid']
        
        train_data = train_data.merge(independent_data, on=['user_id', 'product_id', 'eval_set'], how='inner')
        valid_data = valid_data.merge(independent_data, on=['user_id', 'product_id', 'eval_set'], how='inner')
        
        model = XGBoostModel(self.config, 'level1')
        results = model.train(train_data, valid_data)
        
        model.save_model(self.config.get_model_path('basic_features_xgb.model'))
        
        print(f"Basic Features Model - Train AUC: {results['train_auc']:.4f}, Valid AUC: {results['valid_auc']:.4f}")
    
    def _create_independent_features(self) -> pd.DataFrame:
        user_profile = pd.read_csv(self.config.get_data_path('profile', 'user_profile.csv'))
        product_profile = pd.read_csv(self.config.get_data_path('profile', 'product_basic_profile.csv'))
        product_encoding = pd.read_csv(self.config.get_data_path('profile', 'product_brrc_profile.csv'))
        user_encoding = pd.read_csv(self.config.get_data_path('profile', 'user_brrc_profile.csv'))
        
        dependent_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        
        independent_data = dependent_data[['user_id', 'product_id', 'eval_set']].copy()
        
        independent_data = independent_data.merge(user_profile, on='user_id', how='left')
        independent_data = independent_data.merge(product_profile, on='product_id', how='left')
        independent_data = independent_data.merge(product_encoding, on='product_id', how='left')
        independent_data = independent_data.merge(user_encoding, on='user_id', how='left')
        
        independent_data = independent_data.fillna(0.0)
        
        return independent_data
    
    def _train_ensemble_model(self):
        print("🎯 Training ensemble model...")
        
        ensemble = Level1Ensemble(self.config)
        
        word2vec_features = [f'prod_vec_{i}' for i in range(self.config.feature_params['word2vec_dim'])]
        word2vec_features += [f'user_vec_{i}' for i in range(self.config.feature_params['word2vec_dim'])]
        
        basic_features = [col for col in self._create_independent_features().columns 
                         if col not in ['user_id', 'product_id', 'eval_set']]
        
        ensemble.add_model('word2vec', word2vec_features)
        ensemble.add_model('basic', basic_features)
        
        train_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n_2.csv'))
        valid_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n_1.csv'))
        
        results = ensemble.train_all(train_data, valid_data)
        ensemble.save_all(self.config.get_model_path())
        
        print("Ensemble models trained successfully!")
    
    def train_level2_model(self):
        print("🚀 Training Level 2 model...")
        
        dependent_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        independent_data = self._create_level2_features()
        
        level2_model = Level2Model(self.config)
        results = level2_model.train(dependent_data, independent_data)
        
        level2_model.save_model(self.config.get_model_path('level2_xgb.model'))
        
        print(f"Level 2 Model - Train AUC: {results['train_auc']:.4f}, Valid AUC: {results['valid_auc']:.4f}")
    
    def _create_level2_features(self) -> pd.DataFrame:
        dependent_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        
        independent_data = dependent_data[['user_id', 'product_id', 'eval_set']].copy()
        
        user_profile = pd.read_csv(self.config.get_data_path('profile', 'user_profile.csv'))
        product_profile = pd.read_csv(self.config.get_data_path('profile', 'product_basic_profile.csv'))
        product_encoding = pd.read_csv(self.config.get_data_path('profile', 'product_brrc_profile.csv'))
        user_encoding = pd.read_csv(self.config.get_data_path('profile', 'user_brrc_profile.csv'))
        
        independent_data = independent_data.merge(user_profile, on='user_id', how='left')
        independent_data = independent_data.merge(product_profile, on='product_id', how='left')
        independent_data = independent_data.merge(product_encoding, on='product_id', how='left')
        independent_data = independent_data.merge(user_encoding, on='user_id', how='left')
        
        independent_data = independent_data.fillna(0.0)
        
        return independent_data
    
    def generate_predictions(self, model_type: str = 'level2'):
        print(f"🔮 Generating predictions using {model_type} model...")
        
        if model_type == 'level2':
            self._generate_level2_predictions()
        elif model_type == 'level1':
            self._generate_level1_predictions()
        else:
            raise ValueError("model_type must be 'level1' or 'level2'")
        
        print("✅ Predictions generated!")
    
    def _generate_level2_predictions(self):
        dependent_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        independent_data = self._create_level2_features()
        
        level2_model = Level2Model(self.config)
        level2_model.load_model(self.config.get_model_path('level2_xgb.model'))
        
        predictions = level2_model.predict(dependent_data, independent_data)
        
        submission_data = dependent_data[['user_id', 'product_id']].copy()
        submission_data['predictions'] = predictions
        
        submission = self.submission_generator.create_submission(
            predictions, 
            submission_data['user_id'].values, 
            submission_data['product_id'].values
        )
        
        self.submission_generator.save_submission(
            submission, 
            self.config.get_output_path('level2_submission.csv')
        )
    
    def _generate_level1_predictions(self):
        prod_vecs = pd.read_csv(self.config.get_data_path('profile', 'prodvecs.csv'))
        user_vecs = pd.read_csv(self.config.get_data_path('profile', 'uservecs.csv'))
        
        score_data = pd.read_csv(self.config.get_data_path('model', 'dependent', 'dependent_n.csv'))
        score_data = score_data.merge(prod_vecs, on='product_id', how='inner')
        score_data = score_data.merge(user_vecs, on='user_id', how='inner')
        
        model = XGBoostModel(self.config, 'level1')
        model.load_model(self.config.get_model_path('word2vec_xgb.model'))
        
        predictions = model.predict(score_data)
        
        submission_data = score_data[['user_id', 'product_id']].copy()
        submission_data['predictions'] = predictions
        
        submission = self.submission_generator.create_submission(
            predictions, 
            submission_data['user_id'].values, 
            submission_data['product_id'].values
        )
        
        self.submission_generator.save_submission(
            submission, 
            self.config.get_output_path('level1_submission.csv')
        )
    
    def run_full_pipeline(self):
        print("🎯 Running full Instacart Market Basket Analysis pipeline...")
        
        if not self.validate_setup():
            print("❌ Setup validation failed. Please check your data files.")
            return
        
        self.preprocess_data()
        self.create_features()
        self.create_target_variables()
        self.train_level1_models()
        self.train_level2_model()
        self.generate_predictions('level2')
        
        print("🎉 Full pipeline completed successfully!")
    
    def run_quick_pipeline(self):
        print("⚡ Running quick pipeline for testing...")
        
        if not self.validate_setup():
            print("❌ Setup validation failed. Please check your data files.")
            return
        
        self.preprocess_data()
        self.create_features()
        self.create_target_variables()
        self._train_word2vec_model()
        self._generate_level1_predictions()
        
        print("🎉 Quick pipeline completed successfully!")
