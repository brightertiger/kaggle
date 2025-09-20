import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

class BERTEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        for name, parameter in self.bert.named_parameters():
            if any(layer in name for layer in ['embeddings', 'layer.0', 'layer.1', 'layer.2', 
                                               'layer.3', 'layer.4', 'layer.5', 'layer.6', 
                                               'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11']):
                parameter.requires_grad = False
    
    def forward(self, tokens):
        params = {
            'input_ids': tokens.long(),
            'attention_mask': (tokens > 0).long(),
            'token_type_ids': None,
            'output_all_encoded_layers': False
        }
        state, _ = self.bert(**params)
        return state

class PronounResolutionModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.bert_encoder = BERTEncoder(model_name)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(10, 20)
        
        feature_dim = hidden_size * 3 + 20 + 6
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 150),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(150, 1)
        )
    
    def forward(self, tokens, offsets, feature_a, feature_b):
        batch_size = tokens.shape[0]
        
        bert_output = self.bert_encoder(tokens)
        
        noun_a = bert_output.gather(1, offsets.unsqueeze(2)[:, 0, :].unsqueeze(2).expand(-1, -1, self.hidden_size))
        noun_b = bert_output.gather(1, offsets.unsqueeze(2)[:, 1, :].unsqueeze(2).expand(-1, -1, self.hidden_size))
        pronoun = bert_output.gather(1, offsets.unsqueeze(2)[:, 2, :].unsqueeze(2).expand(-1, -1, self.hidden_size))
        
        noun_a = noun_a.squeeze()
        noun_b = noun_b.squeeze()
        pronoun = pronoun.squeeze()
        
        bert_dist_a = noun_a * pronoun
        bert_dist_b = noun_b * pronoun
        
        act_dist_a = self.embedding(feature_a[:, 0].long())
        act_dist_b = self.embedding(feature_b[:, 0].long())
        
        feats_a = feature_a[:, 1:].view(batch_size, -1)
        feats_b = feature_b[:, 1:].view(batch_size, -1)
        
        feature_a_combined = torch.cat([noun_a, pronoun, bert_dist_a, act_dist_a, feats_a], dim=1)
        feature_b_combined = torch.cat([noun_b, pronoun, bert_dist_b, act_dist_b, feats_b], dim=1)
        
        output_a = self.classifier(feature_a_combined)
        output_b = self.classifier(feature_b_combined)
        output_neither = torch.zeros_like(output_a)
        
        output = torch.cat([output_a, output_b, output_neither], dim=-1)
        return output
