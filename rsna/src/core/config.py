"""Configuration with paths derived at access time, without import-time writes."""
from pathlib import Path
import torch


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
    PRETRAINED = True
    USE_AMP = True
    USE_TTA = False
    TRAIN_LABELS = 'stage_1_train.csv'
    SAMPLE_SUBMISSION = 'stage_1_sample_submission.csv'
    TRAIN_IMAGES = 'train'
    TEST_IMAGES = 'test'
    CLASS_NAMES = ['any', 'epidural', 'intraparenchymal',
                   'intraventricular', 'subarachnoid', 'subdural']
    WINDOW_CENTERS = [40, 80, 40]
    WINDOW_WIDTHS = [80, 200, 380]

    def __init__(self, data_dir='data', output_dir='output'):
        self.DATA_DIR = Path(data_dir)
        self.OUTPUT_DIR = Path(output_dir)
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @property
    def TRAIN_DIR(self):
        return Path(self.DATA_DIR) / self.TRAIN_IMAGES

    @property
    def TEST_DIR(self):
        return Path(self.DATA_DIR) / self.TEST_IMAGES

    @property
    def MODEL_DIR(self):
        return Path(self.OUTPUT_DIR) / 'models'

    @property
    def SCORE_DIR(self):
        return Path(self.OUTPUT_DIR) / 'scores'

    @property
    def TRAIN_CSV(self):
        return Path(self.OUTPUT_DIR) / 'train.csv'

    @property
    def TEST_CSV(self):
        return Path(self.OUTPUT_DIR) / 'test.csv'

    def checkpoint_dir(self, model_name, fold_idx):
        return self.MODEL_DIR / model_name / f'fold_{fold_idx}'

    def ensure_output_dirs(self):
        for directory in (self.OUTPUT_DIR, self.MODEL_DIR, self.SCORE_DIR):
            Path(directory).mkdir(parents=True, exist_ok=True)
