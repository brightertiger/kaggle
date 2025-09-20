import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    # Data paths
    DATA_ROOT: str = "../../data"
    PRETRAIN_DATA_PATH: str = "../../data/pretrain"
    TRAIN_DATA_PATH: str = "../../data/train"
    MODEL_SAVE_PATH: str = "../../model"
    
    # Image processing
    IMAGE_SIZE: int = 256
    LARGE_IMAGE_SIZE: int = 330
    IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Training parameters
    BATCH_SIZE: int = 20
    VALIDATION_BATCH_SIZE: int = 6
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS_PRETRAIN: int = 10
    NUM_EPOCHS_TRAIN: int = 12
    NUM_EPOCHS_COMBINE: int = 10
    
    # Model parameters
    MODEL_NAME: str = "efficientnet-b5"
    DROPOUT_RATE: float = 0.3
    
    # Loss function parameters
    MSE_WEIGHT: float = 0.75
    VARIANCE_WEIGHT: float = 0.2
    LABEL_NOISE_SCALE: float = 0.05
    
    # Cross-validation
    PRETRAIN_FOLDS: int = 10
    TRAIN_FOLDS: int = 5
    RANDOM_SEED: int = 2017
    
    # Data augmentation
    COLOR_JITTER_BRIGHTNESS: float = 0.5
    COLOR_JITTER_CONTRAST: float = 0.3
    COLOR_JITTER_SATURATION: float = 0.3
    SCALE_RANGE: Tuple[float, float] = (1.0, 1.25)
    
    # Hardware
    DEVICE: str = "cuda:0"
    NUM_WORKERS: int = 6
    
    # File names
    TRAIN_LABELS_2015: str = "trainLabels15.csv"
    TEST_LABELS_2015: str = "testLabels15.csv"
    TRAIN_LABELS_2019: str = "trainLabels19.csv"
    TRAIN_FOLDS_FILE: str = "train_folds.csv"
    TEST_FOLDS_FILE: str = "test_folds.csv"
    
    def __post_init__(self):
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(self.MODEL_SAVE_PATH, "pretrain"), exist_ok=True)
        os.makedirs(os.path.join(self.MODEL_SAVE_PATH, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.MODEL_SAVE_PATH, "combine"), exist_ok=True)
