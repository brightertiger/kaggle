import torch 
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

class BCELoss(nn.Module):
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        return None

    def forward(self, logits, target):
        logits = logits.float()
        target = target.float()
        logprobs = F.log_softmax(logits, dim = -1)
        nll_loss = -logprobs * target
        nll_loss = nll_loss.sum(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()