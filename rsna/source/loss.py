import torch 
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = BCEWithLogitsLoss(reduction='none')
        return None
        
    def forward(self, logit, label):
        weight = torch.ones_like(logit)
        weight[:,0] += 1 
        loss = F.binary_cross_entropy_with_logits(logit, label, weight=weight, reduction='none')
        return loss