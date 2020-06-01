import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class DiceLoss(nn.Module):

    def __init__(self, dice_weight, bce_weight):
        super().__init__()
        self.dice_weight = dice_weight 
        self.bce_weight = bce_weight 
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        return None 
    
    def dice_loss(self, scores, actuals):
        smooth = 1e-5
        scores = scores.contiguous().view(-1)
        actuals = actuals.contiguous().view(-1)
        intersection = (scores * actuals).sum() + smooth 
        union = (actuals.sum() + scores.sum() + smooth)
        loss = 1 - (2 * intersection / union)
        return loss

    def forward(self, scores, actuals):
        scores = self.sigmoid(scores)
        bce_loss = self.bce_loss(scores, actuals)
        dce_loss = self.dice_loss(scores, actuals)
        loss = self.bce_weight * bce_loss + self.dice_weight * dce_loss 
        return loss