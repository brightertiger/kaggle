import torch 
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

class BCELoss(nn.Module):
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.positive = torch.tensor([10.]).cuda()
        self.loss = BCEWithLogitsLoss(reduction='none', pos_weight=self.positive)
        return None

    def forward(self, logits, target):
        logits = logits.float()
        target = target.float()
        loss = self.loss(logits, target)
        return loss.mean()