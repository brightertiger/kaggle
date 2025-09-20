"""
Data splitting script for Avito competition.

This script creates cross-validation folds for model training and validation.
Refactored for better organization and maintainability.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

def create_cv_folds():
    """Create cross-validation folds for training data."""
    
    # Load training data
    data = pd.read_csv('../data/download/train.csv')
    data = data.drop(['image'], axis=1)
    data['activation_date'] = pd.to_datetime(data['activation_date'])
    print(f'Training data shape: {data.shape}')
    
    # Create output directory
    os.makedirs('../data/data/files', exist_ok=True)
    
    # Initialize KFold
    folds = KFold(n_splits=5, shuffle=True, random_state=2017)
    
    fold = 1
    for train_idx, test_idx in folds.split(data):
        train = data.iloc[train_idx, :][['item_id']]
        valid = data.iloc[test_idx, :][['item_id']]
        print(f'Fold {fold} - Train: {train.shape}, Valid: {valid.shape}')
        
        train.to_csv(f'../data/data/files/train_{fold}.csv', index=False)
        valid.to_csv(f'../data/data/files/valid_{fold}.csv', index=False)
        fold += 1
    
    # Create test data file
    test_data = pd.read_csv('../data/download/test.csv')[['item_id']]
    test_data.to_csv('../data/data/files/score.csv', index=False)
    print(f'Test data shape: {test_data.shape}')
    
    print("Cross-validation folds created successfully!")

if __name__ == "__main__":
    create_cv_folds()