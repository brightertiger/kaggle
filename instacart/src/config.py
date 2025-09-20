#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    def __init__(self):
        self.data_path = '../data'
        self.output_path = '../output'
        self.model_path = '../models'
        
        self.random_seed = 108
        
        self.train_ratio = 0.8
        self.valid_ratio = 0.2
        
        self.xgb_params = {
            'booster': 'gbtree',
            'nthread': 6,
            'eta': 0.1,
            'max_depth': 12,
            'subsample': 0.75,
            'colsample_bytree': 1.0,
            'colsample_bylevel': 0.9,
            'objective': 'binary:logistic',
            'base_score': 0.10,
            'eval_metric': 'auc',
            'seed': self.random_seed
        }
        
        self.xgb_training_params = {
            'num_boost_round': 400,
            'verbose_eval': 100
        }
        
        self.level2_params = {
            'booster': 'gbtree',
            'nthread': 63,
            'max_depth': 10,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'colsample_bylevel': 0.9,
            'lambda': 1.0,
            'alpha': 0.0,
            'objective': 'binary:logistic',
            'eval_metric': ['logloss'],
            'base_score': 0.1,
            'seed': self.random_seed
        }
        
        self.level2_training_params = {
            'num_boost_round': 2000,
            'early_stopping_rounds': 10,
            'verbose_eval': 150
        }
        
        self.feature_params = {
            'min_products_per_user': 2,
            'min_orders_per_user': 2,
            'tfidf_max_features': 1000,
            'word2vec_dim': 100,
            'word2vec_window': 5,
            'word2vec_min_count': 5
        }
        
        self.encoding_params = {
            'prior_probability': 0.10,
            'smoothing_factor': 100
        }
    
    def get_data_path(self, *paths) -> str:
        return os.path.join(self.data_path, *paths)
    
    def get_output_path(self, *paths) -> str:
        return os.path.join(self.output_path, *paths)
    
    def get_model_path(self, *paths) -> str:
        return os.path.join(self.model_path, *paths)
    
    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_path': self.data_path,
            'output_path': self.output_path,
            'model_path': self.model_path,
            'random_seed': self.random_seed,
            'train_ratio': self.train_ratio,
            'valid_ratio': self.valid_ratio,
            'xgb_params': self.xgb_params,
            'xgb_training_params': self.xgb_training_params,
            'level2_params': self.level2_params,
            'level2_training_params': self.level2_training_params,
            'feature_params': self.feature_params,
            'encoding_params': self.encoding_params
        }
    
    def __str__(self) -> str:
        config_str = "Instacart Market Basket Analysis Configuration:\n"
        config_str += f"  Data Path: {self.data_path}\n"
        config_str += f"  Output Path: {self.output_path}\n"
        config_str += f"  Model Path: {self.model_path}\n"
        config_str += f"  Random Seed: {self.random_seed}\n"
        config_str += f"  Train/Valid Ratio: {self.train_ratio}/{self.valid_ratio}\n"
        return config_str
