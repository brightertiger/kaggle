import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

class Bert(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = BertModel.from_pretrained(model)
        self.__freeze__()
        return None

    def __freeze__(self):
        for name, parameter in self.model.named_parameters():
            if 'embeddings' in name: parameter.requires_grad = False
            if 'layer.0' in name: parameter.requires_grad = False
            if 'layer.1' in name: parameter.requires_grad = False
            if 'layer.2' in name: parameter.requires_grad = False
            if 'layer.3' in name: parameter.requires_grad = False
            if 'layer.4' in name: parameter.requires_grad = False
            if 'layer.5' in name: parameter.requires_grad = False
            if 'layer.6' in name: parameter.requires_grad = False
            if 'layer.7' in name: parameter.requires_grad = False
            if 'layer.8' in name: parameter.requires_grad = False
            if 'layer.9' in name: parameter.requires_grad = False
            if 'layer.10' in name: parameter.requires_grad = False
            if 'layer.11' in name: parameter.requires_grad = False
        return None

    def forward(self, tokens):
        params = {}
        params['input_ids'] = (tokens).long()
        params['attention_mask'] = (tokens > 0).long()
        params['token_type_ids'] = None
        params['output_all_encoded_layers'] = False
        state, _ = self.model(**params)
        return state

class Model(nn.Module):

    def __init__(self, model, size):
        super().__init__()
        self.bert = Bert(model)
        self.size = size
        self.embed = nn.Embedding(10,20)
        head = []
        head += [nn.BatchNorm1d(self.size * 3 + 20 + 6)]
        head += [nn.Dropout(0.2)]
        head += [nn.Linear(self.size * 3 + 20 + 6, 150)]
        head += [nn.ReLU()]
        head += [nn.Dropout(0.2)]
        head += [nn.Linear(150, 1)]
        self.head = nn.Sequential(*head)
        return None

    def forward(self, tokens, offsets, feature_a, feature_b):
        batch = tokens.shape[0]
        bert = self.bert(tokens)
        noun_a = bert.gather(1, offsets.unsqueeze(2)[:, 0, :].unsqueeze(2).expand(-1, -1, self.size))
        noun_b = bert.gather(1, offsets.unsqueeze(2)[:, 1, :].unsqueeze(2).expand(-1, -1, self.size))
        prnoun = bert.gather(1, offsets.unsqueeze(2)[:, 2, :].unsqueeze(2).expand(-1, -1, self.size))
        noun_a = noun_a.squeeze()
        noun_b = noun_b.squeeze()
        prnoun = prnoun.squeeze()
        bert_dist_a = noun_a * prnoun
        bert_dist_b = noun_b * prnoun
        act_dist_a = self.embed(feature_a[:,0].long())
        act_dist_b = self.embed(feature_b[:,0].long())
        feats_a = feature_a[:,1:].view(batch,-1)
        feats_b = feature_b[:,1:].view(batch,-1)
        feature_a = torch.cat([noun_a, prnoun, bert_dist_a, act_dist_a, feats_a], dim=1)
        feature_b = torch.cat([noun_b, prnoun, bert_dist_b, act_dist_b, feats_b], dim=1)
        output_a = self.head(feature_a)
        output_b = self.head(feature_b)
        output_n = torch.zeros_like(output_a)
        output = torch.cat([output_a, output_b, output_n], dim=-1)
        return output

