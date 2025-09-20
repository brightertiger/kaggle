import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AudioConfig:
    sample_rate: int = 32000
    n_mels: int = 300
    fmin: int = 0
    fmax: Optional[int] = None
    segment_length: int = 5
    overlap_ratio: float = 0.75

@dataclass
class ModelConfig:
    num_classes: int = 24
    input_size: int = 300
    pretrained: bool = True
    dropout: float = 0.5

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
    audio: AudioConfig = AudioConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    seed: int = 2017
    device: str = "cuda:0"
    
    def __post_init__(self):
        os.makedirs(self.data.model_save_path, exist_ok=True)
        os.makedirs(self.data.predictions_path, exist_ok=True)
