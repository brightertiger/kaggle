import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from typing import Tuple
from .config import Config

class TweetSentimentModel(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        roberta_config = RobertaConfig.from_pretrained(
            config.data.vocab_file.replace('vocab.json', 'config.json'),
            output_hidden_states=True
        )
        
        self.encoder = RobertaModel.from_pretrained(
            config.data.vocab_file.replace('vocab.json', 'pytorch_model.bin'),
            config=roberta_config
        )
        
        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.classifier = nn.Linear(roberta_config.hidden_size, 3)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, tokens: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.encoder(tokens, attention_mask=masks)
        hidden_states = outputs.hidden_states
        
        features = torch.stack([
            hidden_states[-1], 
            hidden_states[-2], 
            hidden_states[-3], 
            hidden_states[-4]
        ])
        features = torch.mean(features, dim=0)
        features = self.dropout(features)
        
        logits = self.classifier(features)
        start_logits, end_logits, aux_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1), aux_logits.squeeze(-1)

class SoftArgmax(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smax = self.softmax(x)
        indices = torch.arange(x.size(1), device=x.device, dtype=torch.float)
        return torch.matmul(smax, indices)

class TweetLoss(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.dice_loss = DiceLoss()
        self.soft_argmax = SoftArgmax()
    
    def forward(
        self, 
        start_logits: torch.Tensor, 
        end_logits: torch.Tensor, 
        start_idx: torch.Tensor, 
        end_idx: torch.Tensor,
        aux_logits: torch.Tensor,
        aux_labels: torch.Tensor
    ) -> torch.Tensor:
        
        start_loss = self.ce_loss(start_logits, start_idx)
        end_loss = self.ce_loss(end_logits, end_idx)
        ce_loss = start_loss + end_loss
        
        dice_loss = self.dice_loss(aux_logits, aux_labels)
        
        return ce_loss

class DiceLoss(nn.Module):
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        preds_flat = preds.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (preds_flat * target_flat).sum()
        pred_sum = torch.sum(preds_flat * preds_flat)
        target_sum = torch.sum(target_flat * target_flat)
        
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice
