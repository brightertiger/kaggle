import os
from pathlib import Path

class Config:
    SEED = 2017
    IMAGE_SIZE = 512
    BATCH_SIZE_TRAIN = 12
    BATCH_SIZE_VALID = 6
    BATCH_SIZE_INFERENCE = 18
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_FOLDS = 5
    NUM_CLASSES = 6
    NUM_WORKERS = 6
    
    DATA_DIR = Path("../data")
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    OUTPUT_DIR = Path("../output")
    MODEL_DIR = Path("../model")
    SCORE_DIR = Path("../score")
    
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV = DATA_DIR / "test.csv"
    
    CLASS_NAMES = [
        'any', 'epidural', 'intraparenchymal', 
        'intraventricular', 'subarachnoid', 'subdural'
    ]
    
    WINDOW_CENTERS = [40, 80, 40]
    WINDOW_WIDTHS = [80, 200, 380]
    
    DEVICE = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    def __init__(self):
        for directory in [self.DATA_DIR, self.OUTPUT_DIR, self.MODEL_DIR, self.SCORE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
