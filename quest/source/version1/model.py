import torch 
import torch.nn as nn
from transformers import BertModel

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768 * 4, 30)
        return None
    
    def forward(self, question, answer):
        params = {}
        params['input_ids'] = question.long()
        params['attention_mask'] = (question > 0).long().squeeze()
        qseq, qpool = self.bert(**params)
        params = {}
        params['input_ids'] = answer.long()
        params['attention_mask'] = (answer > 0).long().squeeze()
        aseq, apool = self.bert(**params)
        qseq = qseq.mean(dim=1).squeeze()
        aseq = aseq.mean(dim=1).squeeze()
        features = torch.cat([qpool, qseq, apool, aseq], dim=-1)
        output = self.linear(features)
        return output