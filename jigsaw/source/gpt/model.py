import torch
import torch.nn as nn
from pytorch_pretrained_bert import GPT2Model

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768 * 2, 1)
        return None
        
    def forward(self, input_ids):
        states, _ = self.transformer(input_ids, None, None, None)
        avg_pool = torch.mean(states, 1)
        max_pool, _ = torch.max(states, 1)
        concat = torch.cat((avg_pool, max_pool), 1)
        concat = self.dropout(concat)
        logits = self.linear(concat)
        return logits