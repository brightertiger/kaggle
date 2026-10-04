"""Configuration for the original EfficientNet training recipe."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    DATA_ROOT: str = str(Path(__file__).resolve().parents[1] / 'data')
    PRETRAIN_DATA_PATH: str | None = None
    TRAIN_DATA_PATH: str | None = None
    MODEL_SAVE_PATH: str = str(Path(__file__).resolve().parents[1] / 'model')
    IMAGE_SIZE: int = 256
    LARGE_IMAGE_SIZE: int = 330
    IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    BATCH_SIZE: int = 20
    VALIDATION_BATCH_SIZE: int = 6
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS_PRETRAIN: int = 10
    NUM_EPOCHS_TRAIN: int = 12
    NUM_EPOCHS_COMBINE: int = 10
    MODEL_NAME: str = 'efficientnet-b5'
    PRETRAINED: bool = True
    DROPOUT_RATE: float = 0.3
    MSE_WEIGHT: float = 0.75
    VARIANCE_WEIGHT: float = 0.2
    LABEL_NOISE_SCALE: float = 0.05
    USE_NOISE_AUGMENTATION: bool = False
    PRETRAIN_FOLDS: int = 10
    TRAIN_FOLDS: int = 5
    RANDOM_SEED: int = 2017
    COLOR_JITTER_BRIGHTNESS: float = 0.5
    COLOR_JITTER_CONTRAST: float = 0.3
    COLOR_JITTER_SATURATION: float = 0.3
    SCALE_RANGE: Tuple[float, float] = (1.0, 1.25)
    DEVICE: str = 'cuda:0'
    NUM_WORKERS: int = 6
    USE_APEX: bool = True
    TRAIN_LABELS_2015: str = 'trainLabels15.csv'
    TEST_LABELS_2015: str = 'testLabels15.csv'
    TRAIN_LABELS_2019: str = 'train.csv'
    TRAIN_FOLDS_FILE: str = 'train_folds.csv'
    TEST_FOLDS_FILE: str = 'test_folds.csv'

    def __post_init__(self):
        # Constructing a config must not create directories or download weights.
        if self.PRETRAIN_DATA_PATH is None:
            self.PRETRAIN_DATA_PATH = str(Path(self.DATA_ROOT) / 'pretrain')
        if self.TRAIN_DATA_PATH is None:
            self.TRAIN_DATA_PATH = str(Path(self.DATA_ROOT) / 'train')
