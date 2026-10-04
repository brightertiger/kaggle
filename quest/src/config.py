import json
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch


@dataclass
class Config:
    # Model Configuration
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    num_labels: int = 30
    hidden_dropout_prob: float = 0.1
    model_type: str = "question_understanding"
    pretrained: bool = True
    bert_config: Optional[dict] = None
    tokenizer_name: Optional[str] = None
    local_files_only: bool = False
    use_apex: bool = True
    seed: int = 2017

    # Training Configuration
    batch_size: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    num_epochs: int = 6
    gradient_accumulation_steps: int = 4

    # Data Configuration
    text_columns: List[str] = None
    label_columns: List[str] = None
    user_columns: List[str] = None
    url_columns: List[str] = None
    categorical_columns: List[str] = None

    # Cross-validation
    n_folds: int = 5

    # Paths
    data_dir: str = "data/"
    model_dir: str = "models/"
    output_dir: str = "outputs/"

    # Device
    device: str = "auto"

    def __post_init__(self):
        if self.text_columns is None:
            self.text_columns = ['question_title', 'question_body', 'answer']

        if self.label_columns is None:
            self.label_columns = [
                'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
                'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
                'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
                'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
                'question_type_compare', 'question_type_consequence', 'question_type_definition',
                'question_type_entity', 'question_type_instructions', 'question_type_procedure',
                'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
                'answer_helpful', 'answer_level_of_information', 'answer_plausible',
                'answer_relevance', 'answer_satisfaction', 'answer_type_instructions',
                'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written'
            ]

        if self.user_columns is None:
            self.user_columns = ['question_user_name', 'answer_user_name']

        if self.url_columns is None:
            self.url_columns = ['url', 'question_user_page', 'answer_user_page']

        if self.categorical_columns is None:
            self.categorical_columns = ['category', 'host']

        if self.num_labels != len(self.label_columns):
            raise ValueError("num_labels must match label_columns")
        for name in ('batch_size', 'num_epochs', 'gradient_accumulation_steps'):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_folds < 2 or self.max_length < 3:
            raise ValueError("n_folds must be >= 2 and max_length >= 3")

    def resolve_device(self):
        if self.device == 'auto':
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(self.device)

    def load_config(self, config_path: str):
        """Load JSON overrides, rejecting unknown keys and invalid settings."""
        values = json.loads(Path(config_path).read_text(encoding='utf-8'))
        updated = Config(**{**asdict(self), **values})
        self.__dict__.update(updated.__dict__)

    def save_config(self, config_path: str):
        """Save a round-trippable JSON configuration."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf-8')
