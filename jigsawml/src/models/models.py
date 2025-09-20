import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from ..utils.config import Config

class XLMRobertaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(2048, 1)
    
    def forward(self, tokens, attention_mask):
        outputs = self.xlmr(
            input_ids=tokens.long(),
            attention_mask=attention_mask.long()
        )
        
        features = outputs[0]
        cls_token = features[:, 0, :]
        avg_pooled = features.mean(dim=1)
        
        combined_features = torch.cat([cls_token, avg_pooled], dim=-1)
        features = self.dropout(combined_features)
        output = self.output(features)
        
        return output
