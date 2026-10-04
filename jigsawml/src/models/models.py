import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from ..utils.config import Config


class XLMRobertaClassifier(nn.Module):
    def __init__(self, config=None, backbone_config=None):
        super().__init__()
        config = config or Config()
        if backbone_config is not None:
            self.xlmr = XLMRobertaModel(XLMRobertaConfig.from_dict(backbone_config))
        elif config.TINY:
            self.xlmr = XLMRobertaModel(XLMRobertaConfig(
                vocab_size=config.VOCAB_SIZE, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=64,
                max_position_embeddings=config.MAX_LENGTH + 2, pad_token_id=1,
            ))
        elif config.PRETRAINED:
            self.xlmr = XLMRobertaModel.from_pretrained(config.MODEL_NAME)
        else:
            self.xlmr = XLMRobertaModel(XLMRobertaConfig.from_pretrained(config.MODEL_NAME))
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(self.xlmr.config.hidden_size * 2, 1)

    def forward(self, tokens, attention_mask):
        features = self.xlmr(input_ids=tokens.long(), attention_mask=attention_mask.long())[0]
        cls_token = features[:, 0, :]
        # Preserve the original CLS + sequence-mean pooling architecture.
        avg_pooled = features.mean(dim=1)
        return self.output(self.dropout(torch.cat([cls_token, avg_pooled], dim=-1)))
