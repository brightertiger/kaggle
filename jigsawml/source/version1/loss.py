import torch
import torch.nn as nn

class WeightedLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        return None
    
    def forward(self, preds, label, weight):
        loss = self.loss(preds, label) * weight
        return loss