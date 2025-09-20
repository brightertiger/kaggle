import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

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
    
@dataclass
class DataConfig:
    train_path: str = 'data/gap-test.tsv'
    val_path: str = 'data/gap-validation.tsv'
    test_path: str = 'data/gap-development.tsv'
    output_dir: str = 'outputs'
    n_folds: int = 5
    
@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    device: str = 'cuda:0'
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            device=config_dict.get('device', 'cuda:0')
        )
    
    @classmethod
    def default(cls) -> 'Config':
        return cls(
            model=ModelConfig(),
            data=DataConfig()
        )
