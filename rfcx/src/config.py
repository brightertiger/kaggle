from dataclasses import dataclass, field
from typing import Optional

import torch

@dataclass
class AudioConfig:
    sample_rate: int = 32000
    n_mels: int = 300
    fmin: int = 0
    fmax: Optional[int] = None
    segment_length: float = 5
    overlap_ratio: float = 0.75

@dataclass
class ModelConfig:
    num_classes: int = 24
    input_size: int = 300
    pretrained: bool = True
    # Override only for smoke tests; production keeps the selected Res2Net/ResNeSt.
    backbone: Optional[str] = None

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 15
    num_folds: int = 5
    patience: int = 0
    factor: float = 0.5
    min_lr: float = 1e-5
    num_workers: int = 4

@dataclass
class DataConfig:
    train_data_path: str = "data/positive.csv"
    test_data_path: str = "data/sample_submission.csv"
    audio_data_path: str = "data/resample"
    model_save_path: str = "models"
    predictions_path: str = "predictions"

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 2017
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        if self.audio.segment_length <= 0 or self.audio.sample_rate <= 0:
            raise ValueError("Audio duration and sample rate must be positive")
        if not 0 <= self.audio.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
