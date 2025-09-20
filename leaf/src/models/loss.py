import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, preds, targets):
        return self.criterion(preds, targets.squeeze())

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets.squeeze(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=5, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, preds, targets):
        preds = preds.log_softmax(dim=-1)
        true_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (self.classes - 1))
        true_dist.scatter_(1, targets.squeeze().data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * preds, dim=-1))
