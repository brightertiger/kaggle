"""Configuration for the original BERT + linguistic-feature pipeline."""
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    pretrained_model: str = 'bert-large-uncased'
    hidden_size: int = 1024
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 20
    max_length: int = 500
    freeze_layers: int = 12
    random_init: bool = False
    # Used only for random initialization; production loads pretrained BERT.
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 64


@dataclass
class DataConfig:
    train_path: str = 'data/gap-test.tsv'
    val_path: str = 'data/gap-validation.tsv'
    test_path: str = 'data/gap-development.tsv'
    output_dir: str = 'outputs'
    n_folds: int = 5
    num_workers: int = 0
    spacy_model: str = 'en_core_web_lg'


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = 'cuda:0'
    seed: int = 42

    def __post_init__(self):
        if self.data.n_folds < 2:
            raise ValueError('n_folds must be at least 2 for cross-validation')
        if self.model.batch_size < 1 or self.model.epochs < 1:
            raise ValueError('batch_size and epochs must be positive')
        if self.model.max_length < 5:
            raise ValueError('max_length must allow the three mentions and special tokens')

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, encoding='utf-8') as f:
            values = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**values.get('model', {})),
            data=DataConfig(**values.get('data', {})),
            device=values.get('device', 'cuda:0'),
            seed=values.get('seed', 42),
        )

    @classmethod
    def default(cls) -> 'Config':
        return cls()
