import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

class Bert(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 1)
        self.aux = nn.Linear(768, 6)
        return None

    def forward(self, tokens):
        params = {}
        params['input_ids'] = (tokens).long()
        params['attention_mask'] = (tokens > 0).long()
        params['token_type_ids'] = None
        params['output_all_encoded_layers'] = False
        _, state = self.model(**params)
        state = self.dropout(state)
        output = self.output(state)
        aux = self.aux(state)
        return output, aux
