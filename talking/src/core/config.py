import os
from pathlib import Path
from typing import List, Dict, Any

class Config:
    """Configuration class for TalkingData AdTracking Fraud Detection pipeline."""
    
    def __init__(self):
        # Data paths
        self.DATA_DIR = Path("../data")
        self.RAW_DATA_DIR = self.DATA_DIR / "download"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.FEATURES_DIR = self.DATA_DIR / "features"
        self.MODELS_DIR = self.DATA_DIR / "models"
        self.SUBMISSIONS_DIR = self.DATA_DIR / "submissions"
        
        # File names
        self.TRAIN_FILE = "train.csv"
        self.TEST_FILE = "test.csv"
        self.SUPPLEMENT_FILE = "test_supplement.csv"
        
        # Data types for memory optimization
        self.DTYPES = {
            'ip': 'uint32',
            'app': 'uint16', 
            'device': 'uint16',
            'os': 'uint16',
            'channel': 'uint16',
            'is_attributed': 'uint8',
            'click_id': 'uint32'
        }
        
        # Feature engineering parameters
        self.CATEGORICAL_FEATURES = ['ip', 'app', 'channel', 'device', 'os']
        self.TARGET_COLUMN = 'is_attributed'
        
        # Time-based filtering
        self.START_DATE = '2017-11-08 12:00:00'
        self.VALID_DATE = '2017-11-09'
        self.KEEP_HOURS = [4, 5, 9, 10, 13, 14]
        
        # Model parameters
        self.NUM_MODELS = 6
        self.CROSS_VALIDATION_FOLDS = 5
        self.EARLY_STOPPING_ROUNDS = 50
        
        # LightGBM parameters for different models
        self.LGB_PARAMS = {
            'model_1': {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': 0.075,
                'num_leaves': 32,
                'max_depth': -1,
                'min_child_weight': 5,
                'max_bin': 255,
                'subsample': 0.6,
                'subsample_freq': 1,
                'colsample_bytree': 0.3,
                'min_split_gain': 0,
                'scale_pos_weight': 99.7,
                'metric': 'auc',
                'verbose': -1
            },
            'model_2': {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': 0.1,
                'num_leaves': 24,
                'max_depth': -1,
                'min_child_weight': 5,
                'max_bin': 255,
                'subsample': 0.5,
                'subsample_freq': 1,
                'colsample_bytree': 0.3,
                'min_split_gain': 0,
                'scale_pos_weight': 99.7,
                'metric': 'auc',
                'verbose': -1
            }
        }
        
        # Ensemble weights
        self.ENSEMBLE_WEIGHTS = {
            'score_1': 2.0,
            'score_2': 0.5,
            'score_3': 3.0,
            'score_4': 1.0,
            'score_5': 3.0,
            'score_6': 1.5
        }
        
        # System settings
        self.NUM_THREADS = os.cpu_count()
        self.RANDOM_STATE = 42
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DIR,
            self.MODELS_DIR,
            self.SUBMISSIONS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
