"""BERT mention representations and the shared candidate-scoring head."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel


class BERTEncoder(nn.Module):
    def __init__(self, model_name, *, bert_config=None, freeze_layers=12):
        super().__init__()
        self.bert = (BertModel(bert_config) if bert_config is not None
                     else BertModel.from_pretrained(model_name))
        for parameter in self.bert.embeddings.parameters():
            parameter.requires_grad = False
        # Exact module selection avoids accidentally freezing layer.10 through layer.19.
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, tokens):
        return self.bert(input_ids=tokens.long(), attention_mask=tokens.ne(0),
                         return_dict=True).last_hidden_state


class SingletonBatchNorm1d(nn.BatchNorm1d):
    """Keep BatchNorm and use its running statistics for a final singleton batch."""
    def forward(self, inputs):
        if self.training and inputs.shape[0] == 1:
            return F.batch_norm(inputs, self.running_mean, self.running_var,
                                self.weight, self.bias, False, self.momentum, self.eps)
        return super().forward(inputs)


class PronounResolutionModel(nn.Module):
    def __init__(self, model_name, hidden_size, dropout=0.2, *, bert_config=None, freeze_layers=12):
        super().__init__()
        self.bert_encoder = BERTEncoder(model_name, bert_config=bert_config, freeze_layers=freeze_layers)
        self.hidden_size = self.bert_encoder.bert.config.hidden_size
        if hidden_size != self.hidden_size:
            raise ValueError(f'hidden_size={hidden_size} differs from BERT width {self.hidden_size}')
        self.embedding = nn.Embedding(10, 20)
        self.classifier = nn.Sequential(
            SingletonBatchNorm1d(hidden_size * 3 + 20 + 6),
            nn.Dropout(dropout), nn.Linear(hidden_size * 3 + 20 + 6, 150),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(150, 1),
        )

    def forward(self, tokens, offsets, feature_a, feature_b):
        states = self.bert_encoder(tokens)
        mentions = states[torch.arange(tokens.shape[0], device=tokens.device)[:, None], offsets]
        noun_a, noun_b, pronoun = mentions.unbind(dim=1)
        def score(noun, features):
            fused = torch.cat([noun, pronoun, noun * pronoun,
                               self.embedding(features[:, 0].long()), features[:, 1:]], dim=1)
            return self.classifier(fused)
        output_a, output_b = score(noun_a, feature_a), score(noun_b, feature_b)
        return torch.cat([output_a, output_b, torch.zeros_like(output_a)], dim=1)
