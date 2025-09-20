#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from .config import Config


class DataProcessor:
    """Data processing utilities for Jigsaw Toxic Comment Classification."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data."""
        train_path = self.config.get_data_path('', 'train.csv')
        test_path = self.config.get_data_path('', 'test.csv')
        
        print(f"Loading training data from: {train_path}")
        train_data = pd.read_csv(train_path)
        
        print(f"Loading test data from: {test_path}")
        test_data = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        return train_data, test_data
    
    def create_sample_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sample weights for bias mitigation."""
        print("Creating sample weights for bias mitigation...")
        
        weights = data[['id', 'target'] + self.config.identity_columns].copy()
        
        weights['base'] = 1
        
        identity_mask = (weights[self.config.identity_columns].fillna(0).values >= 0.5)
        weights['normal'] = identity_mask.sum(axis=1).astype(bool).astype(int)
        
        toxic_mask = (weights['target'].values >= 0.5)
        non_identity_mask = (weights[self.config.identity_columns].fillna(0).values < 0.5)
        weights['group_1'] = toxic_mask.astype(int) + non_identity_mask.sum(axis=1).astype(bool).astype(int)
        weights['group_1'] = (weights['group_1'] > 1).astype(bool).astype(int)
        
        non_toxic_mask = (weights['target'].values < 0.5)
        weights['group_2'] = non_toxic_mask.astype(int) + identity_mask.sum(axis=1).astype(bool).astype(int)
        weights['group_2'] = (weights['group_2'] > 1).astype(bool).astype(int)
        
        weights['weight'] = weights['base'] + weights['normal'] + weights['group_1'] + weights['group_2']
        weights = weights[['id', 'weight']]
        
        weight_distribution = weights['weight'].value_counts().sort_index()
        print("Weight distribution:")
        print(weight_distribution)
        
        return weights
    
    def prepare_data_for_training(self, data: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for model training."""
        print("Preparing data for training...")
        
        features = ['id', 'comment_text', 'weight']
        labels = ['target']
        aux_labels = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        
        prepared_data = weights.merge(data, on='id')
        prepared_data = prepared_data[features + labels + aux_labels]
        
        prepared_data[labels] = prepared_data[labels].fillna(0.0)
        prepared_data[aux_labels] = prepared_data[aux_labels].fillna(0.0)
        prepared_data['comment_text'] = prepared_data['comment_text'].fillna('none blank')
        
        print(f"Prepared data shape: {prepared_data.shape}")
        return prepared_data
    
    def create_cross_validation_splits(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits."""
        print(f"Creating {self.config.n_folds}-fold cross-validation splits...")
        
        splitter = StratifiedKFold(
            n_splits=self.config.n_folds, 
            random_state=self.config.random_seed, 
            shuffle=True
        )
        
        splits = list(splitter.split(data.index, data['weight']))
        print(f"Created {len(splits)} CV splits")
        
        return splits
    
    def save_fold_data(self, data: pd.DataFrame, splits: List[Tuple[np.ndarray, np.ndarray]]):
        """Save data for each fold."""
        print("Saving fold data...")
        
        for fold_idx, (train_idx, valid_idx) in enumerate(splits, 1):
            train_data = data.iloc[train_idx]
            valid_data = data.iloc[valid_idx]
            
            train_path = self.config.get_data_path('', f'train_fold_{fold_idx}.csv')
            valid_path = self.config.get_data_path('', f'valid_fold_{fold_idx}.csv')
            
            train_data.to_csv(train_path, index=False)
            valid_data.to_csv(valid_path, index=False)
            
            print(f"Fold {fold_idx}: Train {train_data.shape}, Valid {valid_data.shape}")
    
    def process_all_data(self) -> Dict[str, Any]:
        """Process all data and create training splits."""
        print("Starting data processing...")
        
        train_data, test_data = self.load_data()
        
        sample_weights = self.create_sample_weights(train_data)
        weights_path = self.config.get_data_path('', 'sample_weights.csv')
        sample_weights.to_csv(weights_path, index=False)
        print(f"Saved sample weights to: {weights_path}")
        
        prepared_data = self.prepare_data_for_training(train_data, sample_weights)
        
        cv_splits = self.create_cross_validation_splits(prepared_data)
        self.save_fold_data(prepared_data, cv_splits)
        
        test_data['comment_text'] = test_data['comment_text'].fillna('none blank')
        test_path = self.config.get_data_path('', 'test_processed.csv')
        test_data.to_csv(test_path, index=False)
        print(f"Saved processed test data to: {test_path}")
        
        return {
            'train_data': prepared_data,
            'test_data': test_data,
            'sample_weights': sample_weights,
            'cv_splits': cv_splits,
            'n_folds': self.config.n_folds
        }


class DataLoader:
    """Data loading utilities for model training."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_fold_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data for a specific fold."""
        train_path = self.config.get_data_path('', f'train_fold_{fold}.csv')
        valid_path = self.config.get_data_path('', f'valid_fold_{fold}.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            raise FileNotFoundError(f"Fold {fold} data not found. Run data processing first.")
        
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        
        print(f"Loaded fold {fold}: Train {train_data.shape}, Valid {valid_data.shape}")
        return train_data, valid_data
    
    def load_test_data(self) -> pd.DataFrame:
        """Load processed test data."""
        test_path = self.config.get_data_path('', 'test_processed.csv')
        
        if not os.path.exists(test_path):
            raise FileNotFoundError("Processed test data not found. Run data processing first.")
        
        test_data = pd.read_csv(test_path)
        print(f"Loaded test data: {test_data.shape}")
        return test_data
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data statistics for analysis."""
        stats = {
            'total_samples': len(data),
            'target_distribution': data['target'].value_counts().to_dict(),
            'weight_distribution': data['weight'].value_counts().to_dict(),
            'identity_column_stats': {}
        }
        
        for col in self.config.identity_columns:
            if col in data.columns:
                stats['identity_column_stats'][col] = {
                    'count': data[col].notna().sum(),
                    'mean': data[col].mean(),
                    'std': data[col].std()
                }
        
        return stats
