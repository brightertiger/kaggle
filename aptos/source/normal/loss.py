import torch 
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

class BCELoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.loss = BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma

    def forward(self, logit, label):
        target = torch.zeros_like(logit)
        target = target.scatter(1, label.reshape(-1,1), 1)
        target = target.float()
        target = target * 0.9 + 0.02
        loss = self.loss(logit, target)
        loss = loss.sum(dim=-1)
        return loss
    
class CustomLoss(nn.Module):
    
    def __init__(self, weight=1., variance=0.):
        super().__init__()
        self.mse_loss = MSELoss(reduction='none')
        self.bce_loss = BCELoss()
        self.weight = weight
        self.variance = variance
        return None
    
    def forward(self, regress, classify, label, weight):
        regress = torch.clamp(regress, -0.5, 4.5)
        mse_loss = self.mse_loss(regress, label)
        bce_loss = self.bce_loss(classify, label.long())
        loss = (self.weight * mse_loss) + ((1. - self.weight) * bce_loss)
        loss = loss * weight
        return loss
    
        