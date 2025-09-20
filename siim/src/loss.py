import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=Config.LABEL_SMOOTHING):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + self.smoothing / targets.size(-1)
        
        return self.bce_loss(logits, targets)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.7, focal_weight=0.3, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.focal_loss = FocalLoss()
    
    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets.argmax(dim=-1))
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss
