import os
import json
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_path: str = "data/raw/train.csv"
    test_path: str = "data/raw/test.csv"
    output_dir: str = "data/processed"
    n_folds: int = 10
    random_state: int = 2017

    # Text preprocessing variants
    preprocessing_methods: List[str] = None

    def __post_init__(self):
        if self.preprocessing_methods is None:
            self.preprocessing_methods = [
                "basic_clean",
                "basic_clean_lower",
                "tokenized",
                "nltk_tokenized",
                "preprocessed"
            ]


@dataclass
class ModelConfig:
    """Configuration for model training"""
    # Neural network parameters
    seq_length: int = 200
    embed_size: int = 300
    usable_vocab: int = 30000

    # Training parameters
    batch_size: int = 256
    epochs: int = 12
    learning_rate: float = 1e-3
    patience: int = 12

    # Model architecture
    rnn_units: int = 50
    recurrent_dropout: float = 0.2
    dropout: float = 0.1
    dense_units: int = 256

    # Sparse features (lower min_df for tiny smoke-test datasets)
    nb_min_df: int = 3
    nb_max_df: float = 0.9
    word_max_features: int = 10000
    char_max_features: int = 50000

    # Embedding paths
    glove_path: str = "data/embeddings/glove.840B.300d.txt"
    fasttext_path: str = "data/embeddings/fasttext.txt"
    embedding_type: str = "glove"
    random_embeddings: bool = False
    cpu_only: bool = False

    # Output paths
    model_dir: str = "models"
    log_dir: str = "logs"
    submission_dir: str = "submissions"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    target_columns: List[str] = None

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = [
                'toxic', 'severe_toxic', 'obscene',
                'threat', 'insult', 'identity_hate'
            ]


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.data.output_dir,
            self.model.model_dir,
            self.model.log_dir,
            self.model.submission_dir
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)


def get_config(path: str | None = None) -> Config:
    """Load defaults or a JSON object with data/model/evaluation sections."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    unknown = set(values) - {"data", "model", "evaluation"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(
        data=DataConfig(**values.get("data", {})),
        model=ModelConfig(**values.get("model", {})),
        evaluation=EvaluationConfig(**values.get("evaluation", {})),
    )
