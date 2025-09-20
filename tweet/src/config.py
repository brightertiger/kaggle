from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class DataConfig:
    train_path: str = "data/raw/train.csv"
    test_path: str = "data/raw/test.csv"
    processed_path: str = "data/processed/"
    model_path: str = "models/"
    vocab_file: str = "models/pretrain/vocab.json"
    merges_file: str = "models/pretrain/merges.txt"
    max_length: int = 200
    batch_size: int = 4
    num_workers: int = 2
    n_folds: int = 10
    random_seed: int = 2017

@dataclass
class ModelConfig:
    model_name: str = "roberta-base"
    hidden_size: int = 768
    dropout_rate: float = 0.5
    learning_rate: float = 3e-5
    weight_decay: float = 0.001
    max_epochs: int = 5
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 1e-6
    scheduler_patience: int = 0

@dataclass
class TrainingConfig:
    device: str = "cuda:0"
    mixed_precision: bool = False
    save_best_only: bool = True
    early_stopping_patience: int = 3
    log_interval: int = 100

@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    def __post_init__(self):
        os.makedirs(self.data.processed_path, exist_ok=True)
        os.makedirs(self.data.model_path, exist_ok=True)

def get_config() -> Config:
    return Config()
