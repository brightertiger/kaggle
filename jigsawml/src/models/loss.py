import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets, weights):
        loss = self.bce_loss(predictions, targets) * weights
        return loss

def reduce_loss(loss):
    return loss.sum() / loss.shape[0]
