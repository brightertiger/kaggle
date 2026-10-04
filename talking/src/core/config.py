import os
from pathlib import Path

class Config:
    """Configuration class for TalkingData AdTracking Fraud Detection pipeline."""
    
    def __init__(self, data_dir="data", raw_data_dir=None):
        # Data paths
        self.DATA_DIR = Path(data_dir)
        self._raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else None
        
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
        self.NUM_BOOST_ROUND = 1000
        self.EARLY_STOPPING_ROUNDS = 50
        self.LOG_EVALUATION_PERIOD = 50
        
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
        self.NUM_THREADS = os.cpu_count() or 1
        self.RANDOM_STATE = 42

    @property
    def MODEL_NAMES(self):
        return list(self.LGB_PARAMS)

    @property
    def NUM_MODELS(self):
        return len(self.MODEL_NAMES)

    @property
    def RAW_DATA_DIR(self):
        return self._raw_data_dir if self._raw_data_dir is not None else Path(self.DATA_DIR) / "download"

    @property
    def PROCESSED_DATA_DIR(self):
        return Path(self.DATA_DIR) / "processed"

    @property
    def FEATURES_DIR(self):
        return Path(self.DATA_DIR) / "features"

    @property
    def MODELS_DIR(self):
        return Path(self.DATA_DIR) / "models"

    @property
    def SUBMISSIONS_DIR(self):
        return Path(self.DATA_DIR) / "submissions"
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            Path(self.DATA_DIR),
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DIR,
            self.MODELS_DIR,
            self.SUBMISSIONS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
