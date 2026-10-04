"""Typed defaults and JSON overrides for the competition pipeline."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


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
    pretrained: bool = True
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50265
    dropout_rate: float = 0.5
    learning_rate: float = 3e-5
    weight_decay: float = 0.001
    max_epochs: int = 5
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 1e-6
    scheduler_patience: int = 0
    # The migrated solution returned CE only; keep that objective by default.
    auxiliary_loss_weight: float = 0.0


@dataclass
class TrainingConfig:
    device: str = "cuda:0"
    mixed_precision: bool = False
    save_best_only: bool = True
    early_stopping_patience: int = 3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        return asdict(self)


def get_config(path=None) -> Config:
    if path is None:
        return Config()
    values = json.loads(Path(path).read_text())
    unknown = set(values) - {'data', 'model', 'training'}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(
        data=DataConfig(**values.get('data', {})),
        model=ModelConfig(**values.get('model', {})),
        training=TrainingConfig(**values.get('training', {})),
    )
