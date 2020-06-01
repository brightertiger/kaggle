import torch
import torch.nn as nn

class Loss(nn.Module):
    
    def __init__(self, weight):
        super().__init__()
        self.BCELoss = nn.BCELoss(reduction='none')
        self.MSELoss = nn.MSELoss(reduction='none')
        self.weight = weight
        return None
    
    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        bce_loss = self.BCELoss(preds, labels)
        mse_loss = self.MSELoss(preds, labels)
        tot_loss = bce_loss + self.weight * mse_loss
        return tot_loss