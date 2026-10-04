from pathlib import Path
import torch


class Config:
    SEED = 2017
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = 'data/process'
    MODEL_DIR = 'model'
    MAX_LENGTH = 300
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS_V1 = 5
    EPOCHS_V2 = 4
    LR_V1 = 1e-5
    LR_V2 = 1e-6
    MODEL_NAME = 'xlm-roberta-large'
    PRETRAINED = True
    TINY = False
    VOCAB_SIZE = 256
    ACCUMULATION_STEPS = 4
    N_FOLDS = 5
    LGB_ROUNDS = 250
    LGB_EARLY_STOPPING = 50
    USE_MODEL = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
    USE_FEATURES = [f'use_{i}' for i in range(512)]
    LGB_PARAMS = {
        'boosting_type': 'gbdt', 'objective': 'binary',
        'learning_rate': 0.02, 'num_leaves': 128, 'max_depth': -1,
        'min_child_weight': 100, 'max_bin': 1024, 'subsample': 0.7,
        'subsample_freq': 1, 'colsample_bytree': 0.5,
        'min_split_gain': 0, 'num_threads': 15, 'verbosity': -1,
        'metric': 'auc', 'seed': SEED,
    }

    def __init__(self, data_dir='data', model_dir='model', **overrides):
        self.DATA_DIR = str(Path(data_dir) / 'process')
        self.MODEL_DIR = str(model_dir)
        self.LGB_PARAMS = dict(type(self).LGB_PARAMS)
        for name, value in overrides.items():
            if not hasattr(self, name):
                raise ValueError(f'Unknown configuration option: {name}')
            setattr(self, name, value)
