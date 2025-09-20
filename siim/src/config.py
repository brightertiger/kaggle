import os
from pathlib import Path

class Config:
    SEED = 42
    DEVICE = 'cuda:0'
    
    # Data paths
    DATA_DIR = Path('data')
    TRAIN_IMAGES_DIR = DATA_DIR / 'train'
    TEST_IMAGES_DIR = DATA_DIR / 'test'
    METADATA_DIR = DATA_DIR / 'metadata'
    
    # Model paths
    MODEL_DIR = Path('models')
    SCORE_DIR = Path('scores')
    
    # Training parameters
    IMAGE_SIZE = 512
    BATCH_SIZE = 10
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.0
    
    # Model architecture
    MODEL_NAME = 'efficientnet-b5'
    NUM_CLASSES = 4
    METADATA_DIM = 13
    
    # Data augmentation
    CUTOUT_HOLES = 16
    CUTOUT_SIZE = 64
    
    # Loss function
    POS_WEIGHT = 4.0
    LABEL_SMOOTHING = 0.1
    
    # Cross-validation
    N_FOLDS = 5
    
    # Lookup tables
    SEX_LOOKUP = {'male': 1, 'female': 2}
    AGE_LOOKUP = {20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6, 80: 7}
    ANATOMY_LOOKUP = {
        'lower extremity': 1, 
        'upper extremity': 2, 
        'torso': 3, 
        'head/neck': 4
    }
    DIAGNOSIS_LOOKUP = {
        'other': 0, 
        'melanoma': 1, 
        'nevus': 2, 
        'keratosis': 3
    }
