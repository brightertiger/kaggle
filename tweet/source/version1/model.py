import torch 
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

PREFIX = '/root/model/pretrain/'
    
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        config = RobertaConfig.from_pretrained(PREFIX + 'config.json', output_hidden_states=True)    
        self.encoder = RobertaModel.from_pretrained(PREFIX + 'pytorch_model.bin', config=config)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(config.hidden_size, 3)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        return None

    def forward(self, tokens, masks):
        _, _, hidden = self.encoder(tokens, masks)
        features = torch.stack([hidden[-1], hidden[-2], hidden[-3], hidden[-4]])
        features = torch.mean(features, 0)
        features = self.dropout(features)
        out = self.linear(features)
        start_logits, end_logits, aux_out = out.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        aux_logits = aux_out.squeeze(-1)
        return start_logits, end_logits, aux_logits