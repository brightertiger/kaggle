import os

class Config:
    RANDOM_STATE = 2017
    IMAGE_SIZE = 75
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 20
    LEARNING_RATE = 1e-4
    
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    SUBMISSION_DIR = 'submissions'
    
    FOLDS = 5
    
    IMAGE_TRANSFORMS = {
        'source_1': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'zoom_range': 0.3,
            'rotation_range': 10
        },
        'source_2': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'zoom_range': 0.2,
            'rotation_range': 5,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1
        }
    }
    
    MODEL_CONFIGS = {
        'cnn_basic': {
            'epochs': 100,
            'patience': 20,
            'steps_per_epoch': 40
        },
        'cnn_advanced': {
            'epochs': 150,
            'patience': 20,
            'steps_per_epoch': 40
        },
        'vgg16': {
            'epochs': 100,
            'patience': 25,
            'steps_per_epoch': 40,
            'fine_tune_epochs': 100
        }
    }
    
    XGBOOST_PARAMS = {
        'booster': 'gbtree',
        'nthread': 6,
        'max_depth': 5,
        'min_child_weight': 4,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.8,
        'lambda': 4.0,
        'alpha': 3.0,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],
        'seed': RANDOM_STATE,
        'eta': 0.01
    }
