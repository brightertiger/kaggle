import os
from pathlib import Path

class Config:
    # Data paths
    DATA_DIR = Path('data')
    TRAIN_FILE = DATA_DIR / 'train.csv'
    TEST_FILE = DATA_DIR / 'test.csv'
    
    # Model paths
    MODEL_DIR = Path('models')
    SCORE_DIR = Path('scores')
    
    # Text preprocessing
    MAX_SEQUENCE_LENGTH = 90
    EMBEDDING_DIM = 50
    GLOVE_PATH = Path('glove/glove.6B.50d.txt')
    
    # Author mapping
    AUTHOR_MAP = {'EAP': 0, 'HPL': 1, 'MWS': 2}
    AUTHOR_NAMES = ['EAP', 'HPL', 'MWS']
    NUM_CLASSES = 3
    
    # XGBoost parameters
    XGB_PARAMS = {
        'booster': 'gbtree',
        'nthread': 7,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.75,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.9,
        'lambda': 2.0,
        'alpha': 1.0,
        'objective': 'multi:softprob',
        'eval_metric': ['mlogloss'],
        'num_class': 3,
        'seed': 2017
    }
    
    # Training parameters
    XGB_NUM_ROUNDS = 500
    XGB_EARLY_STOPPING = 50
    XGB_LEARNING_RATE = 0.05
    
    # Neural network parameters
    NN_BATCH_SIZE = 8
    NN_EPOCHS = 20
    NN_LEARNING_RATE = 0.0001
    NN_VALIDATION_SPLIT = 0.15
    
    # Cross-validation
    N_FOLDS = 5
    RANDOM_STATE = 2017
    
    # Feature engineering
    NGRAM_RANGE_WORD = (1, 3)
    NGRAM_RANGE_CHAR = (1, 5)
    NGRAM_RANGE_CHAR_CNT = (1, 7)
    SVD_COMPONENTS = 10
    
    # File names
    TRAIN_TEXT_FEATS = 'train_text_feats.csv'
    TEST_TEXT_FEATS = 'test_text_feats.csv'
    TRAIN_NB_SCORE = 'train_nb_score.csv'
    TEST_NB_SCORE = 'test_nb_score.csv'
    TRAIN_NB_FEATS = 'train_nb_feats.csv'
    TEST_NB_FEATS = 'test_nb_feats.csv'
    TRAIN_NN_SCORE = 'train_nn_score.csv'
    TEST_NN_SCORE = 'test_nn_score.csv'
    TRAIN_LSTM_SCORE = 'train_lstm_score.csv'
    TEST_LSTM_SCORE = 'test_lstm_score.csv'
    XGB_MODEL = 'xgb_model'
    XGB_SCORE = 'xgb_score.csv'
