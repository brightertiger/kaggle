"""Production defaults; instantiate Config to override settings per run."""
from dataclasses import dataclass
import torch


@dataclass
class Config:
    SEED: int = 42
    NUM_CLASSES: int = 5
    IMAGE_SIZE: int = 512
    BATCH_SIZE: int = 6
    NUM_WORKERS: int = 2
    EPOCHS: int = 20
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-6
    DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME: str = 'tf_efficientnet_b4_ns'
    PRETRAINED: bool = True
    N_FOLDS: int = 5
    ACCUMULATION_STEPS: int = 4
    SWA_START: int = 7  # Zero-based epoch index.
    PATIENCE: int | None = None
    NUM_TTA: int = 1
    DATA_DIR: str = './data'
    OUTPUT_DIR: str = './output'
    VERSIONS: tuple[str, ...] = tuple(f'version{i}' for i in range(8))

    CLASS_NAMES = [
        'Cassava Bacterial Blight (CBB)',
        'Cassava Brown Streak Disease (CBSD)',
        'Cassava Green Mottle (CGM)',
        'Cassava Mosaic Disease (CMD)',
        'Healthy',
    ]

    def __post_init__(self):
        if self.N_FOLDS < 2 or self.NUM_CLASSES < 2:
            raise ValueError('At least two folds and classes are required')
        if min(self.IMAGE_SIZE, self.BATCH_SIZE, self.EPOCHS, self.ACCUMULATION_STEPS) < 1:
            raise ValueError('Image size, batch size, epochs and accumulation must be positive')
        if self.NUM_WORKERS < 0 or self.SWA_START < 0 or not 1 <= self.NUM_TTA <= 8:
            raise ValueError('Invalid worker count, SWA start or TTA count')
        if not self.VERSIONS or len(set(self.VERSIONS)) != len(self.VERSIONS):
            raise ValueError('Version names must be nonempty and unique')
