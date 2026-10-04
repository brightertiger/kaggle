import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Optional


def bert_configuration(config):
    if config.bert_config is not None:
        return BertConfig(**config.bert_config)
    return BertConfig.from_pretrained(
        config.model_name, local_files_only=config.local_files_only
    )


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class QuestionUnderstandingModel(nn.Module):
    """BERT-based model for question understanding and answer quality assessment"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.pretrained:
            self.bert = BertModel.from_pretrained(
                config.model_name, local_files_only=config.local_files_only
            )
        else:
            self.bert = BertModel(bert_configuration(config))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Pooler and masked sequence mean for each of question and answer.
        self.classifier = nn.Linear(self.bert.config.hidden_size * 4, config.num_labels)

    def forward(self, question: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            question: Tokenized question text [batch_size, seq_len]
            answer: Tokenized answer text [batch_size, seq_len]

        Returns:
            Logits for multi-label classification [batch_size, num_labels]
        """
        # Process question
        question_attention_mask = (question != self.bert.config.pad_token_id).long()
        question_outputs = self.bert(
            input_ids=question,
            attention_mask=question_attention_mask
        )
        question_pooler = question_outputs.pooler_output
        question_sequence = masked_mean(question_outputs.last_hidden_state, question_attention_mask)

        # Process answer
        answer_attention_mask = (answer != self.bert.config.pad_token_id).long()
        answer_outputs = self.bert(
            input_ids=answer,
            attention_mask=answer_attention_mask
        )
        answer_pooler = answer_outputs.pooler_output
        answer_sequence = masked_mean(answer_outputs.last_hidden_state, answer_attention_mask)

        # Concatenate features
        combined_features = torch.cat([
            question_pooler,
            question_sequence,
            answer_pooler,
            answer_sequence
        ], dim=-1)

        # Apply dropout and classification
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)

        return logits


class DualBERTModel(nn.Module):
    """Alternative architecture with separate BERT encoders for question and answer"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Separate BERT models for question and answer
        bert_config = bert_configuration(config)
        self.question_bert = BertModel(bert_config)
        self.answer_bert = BertModel(bert_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size * 2, config.num_labels)

    def forward(self, question: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        """Forward pass with separate BERT encoders"""
        # Process question
        question_attention_mask = (question != self.question_bert.config.pad_token_id).long()
        question_outputs = self.question_bert(
            input_ids=question,
            attention_mask=question_attention_mask
        )
        question_features = question_outputs.pooler_output

        # Process answer
        answer_attention_mask = (answer != self.answer_bert.config.pad_token_id).long()
        answer_outputs = self.answer_bert(
            input_ids=answer,
            attention_mask=answer_attention_mask
        )
        answer_features = answer_outputs.pooler_output

        # Combine features
        combined_features = torch.cat([question_features, answer_features], dim=-1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)

        return logits


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""

    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = nn.ModuleList(models)

        if not models:
            raise ValueError("An ensemble needs at least one model")
        weights = [1.0] * len(models) if weights is None else weights
        values = torch.tensor(weights, dtype=torch.float)
        if (len(weights) != len(models) or not torch.isfinite(values).all()
                or (values < 0).any() or values.sum() <= 0):
            raise ValueError("Provide one finite nonnegative weight per model with a positive sum")
        self.register_buffer('weights', values / values.sum())

    def forward(self, question: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        """Ensemble forward pass"""
        outputs = []
        for model in self.models:
            output = model(question, answer)
            outputs.append(output)

        # Weighted average of outputs
        ensemble_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            ensemble_output += weight * output

        return ensemble_output


class ModelFactory:
    """Factory class for creating different model architectures"""

    @staticmethod
    def create_model(model_type: str, config) -> nn.Module:
        """Create a model based on the specified type"""
        if model_type == "question_understanding":
            return QuestionUnderstandingModel(config)
        elif model_type == "dual_bert":
            return DualBERTModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_ensemble(models_config: list, config) -> EnsembleModel:
        """Create an ensemble of models"""
        models = []
        weights = []

        for model_config in models_config:
            model_type = model_config.get('type', 'question_understanding')
            weight = model_config.get('weight', 1.0)

            model = ModelFactory.create_model(model_type, config)
            models.append(model)
            weights.append(weight)

        return EnsembleModel(models, weights)
