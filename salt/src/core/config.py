"""Configuration and paths for the seismic segmentation pipeline."""
from pathlib import Path
import torch


class Config:
    def __init__(self, data_dir=None, output_dir=None):
        root = Path(__file__).resolve().parents[2]
        self.DATA_DIR = Path(data_dir) if data_dir is not None else root / 'data'
        self.OUTPUT_DIR = Path(output_dir) if output_dir is not None else root / 'output'
        self.ORIGINAL_SIZE = 101
        self.IMAGE_SIZE = 101
        self.PADDED_SIZE = 128
        self.BATCH_SIZE_TRAIN = 32
        self.BATCH_SIZE_VALID = 32
        self.NUM_WORKERS = 0
        self.NUM_EPOCHS = 200
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.0001
        self.DROPOUT = 0.25
        self.NUM_FOLDS = 5
        self.EARLY_STOPPING_PATIENCE = 50
        self.LR_REDUCTION_PATIENCE = 10
        self.LR_REDUCTION_FACTOR = 0.5
        self.PRETRAINED = True
        self.TINY_MODEL = False
        self.MODEL_NAME = 'seresnet34'
        self.USE_FLIP_AUGMENTATION = True
        self.FLIP_PROBABILITY = 0.5
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.RANDOM_SEED = 2017
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.LOSS_TYPE = 'lovasz'
        self.DICE_WEIGHT = 1.0
        self.BCE_WEIGHT = 1.0
        self.METRIC_TYPE = 'iou'
        self.IOU_CUTOFF = -0.18  # Logits, not probabilities.
        self.IOU_SQUASH = False
        self.MIN_SALT_PIXELS = 25

    @property
    def RAW_DATA_DIR(self):
        return Path(self.DATA_DIR) / 'download'

    @property
    def PROCESSED_DATA_DIR(self):
        return Path(self.OUTPUT_DIR) / 'processed'

    @property
    def MODEL_TAG(self):
        return self.MODEL_NAME + ('_tiny' if self.TINY_MODEL else '')

    @property
    def MODEL_DIR(self):
        return Path(self.OUTPUT_DIR) / 'models' / self.MODEL_TAG

    @property
    def SCORES_DIR(self):
        return Path(self.OUTPUT_DIR) / 'scores' / self.MODEL_TAG

    @property
    def SUBMIT_DIR(self):
        return Path(self.OUTPUT_DIR) / 'submit'

    @property
    def PAD_BEFORE(self):
        return (self.PADDED_SIZE - self.IMAGE_SIZE + 1) // 2

    def validate(self):
        if self.IMAGE_SIZE < 2 or self.PADDED_SIZE < self.IMAGE_SIZE:
            raise ValueError('Require 2 <= IMAGE_SIZE <= PADDED_SIZE')
        if self.PADDED_SIZE < 32 or self.PADDED_SIZE % 32:
            raise ValueError('PADDED_SIZE must be a positive multiple of 32')
        if self.NUM_FOLDS < 2 or self.NUM_EPOCHS < 1:
            raise ValueError('Require at least two folds and one epoch')
        if min(self.BATCH_SIZE_TRAIN, self.BATCH_SIZE_VALID) < 1:
            raise ValueError('Batch sizes must be positive')

    def _create_directories(self):
        self.validate()
        for directory in (self.PROCESSED_DATA_DIR, self.MODEL_DIR,
                          self.SCORES_DIR / 'valid', self.SCORES_DIR / 'test',
                          self.SUBMIT_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            key = key.upper()
            if key not in self.__dict__:
                raise ValueError(f'Unknown configuration parameter: {key}')
            setattr(self, key, value)

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in self.__dict__.items()}

    def print_config(self):
        for key, value in self.to_dict().items():
            print(f'{key}: {value}')
