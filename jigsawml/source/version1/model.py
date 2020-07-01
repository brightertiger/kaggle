import torch 
import torch.nn as nn
from transformers import XLMRobertaModel

class XLMModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(2048, 1)
        return None
    
    def forward(self, tokens, attens):
        params = {}
        params['input_ids'] = tokens.long()
        params['attention_mask'] = attens.long()
        features, _ = self.xlmr(**params)
        clst = features[:,0,:]
        avgt = features.mean(dim=1).squeeze()
        features = torch.cat([clst, avgt], dim=-1)
        features = self.dropout(features)
        output = self.output(features)
        return output
    