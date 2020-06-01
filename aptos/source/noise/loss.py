import torch 
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

class VarLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = MSELoss()
        return None
    
    def forward(self, score_1, score_2):
        loss = (score_1 - score_2)
        loss = torch.pow(loss,2)
        loss = loss.sum(dim=-1)
        return loss

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
        self.var_loss = VarLoss()
        self.weight = weight
        self.variance = variance
        return None
    
    def forward(self, regress_1, regress_2, classify_1, classify_2, label, weight):
        regress = (regress_1 + regress_2) / 2
        classify = (classify_1 + classify_2) / 2
        regress = torch.clamp(regress, -0.5, 4.5)
        mse_loss = self.mse_loss(regress, label)
        bce_loss = self.bce_loss(classify, label.long())
        var_loss = self.var_loss(regress_1, regress_2) + self.var_loss(classify_1, classify_2)
        loss = (self.weight * mse_loss) + ((1. - self.weight) * bce_loss) + (self.variance * var_loss)
        loss = loss * weight
        return loss
    
        