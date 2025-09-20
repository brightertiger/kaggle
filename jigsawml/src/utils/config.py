import os

class Config:
    SEED = 2017
    DEVICE = 'cuda:0'
    
    # Data paths
    DATA_DIR = '../../data/process'
    MODEL_DIR = '../../model'
    
    # Model parameters
    MAX_LENGTH = 300
    BATCH_SIZE = 8
    NUM_WORKERS = 10
    EPOCHS_V1 = 5
    EPOCHS_V2 = 4
    
    # Learning rates
    LR_V1 = 1e-5
    LR_V2 = 1e-6
    
    # XLM-RoBERTa parameters
    MODEL_NAME = 'xlm-roberta-large'
    
    # LightGBM parameters
    LGB_PARAMS = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.02,
        'num_leaves': 128,
        'max_depth': -1,
        'min_child_weight': 100,
        'max_bin': 1024,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.5,
        'min_split_gain': 0,
        'nthread': 15,
        'verbose': 0,
        'metric': 'auc'
    }
    
    # USE embedding features
    USE_FEATURES = [f'use_{i}' for i in range(512)]
    
    # Cross-validation folds
    N_FOLDS = 5
