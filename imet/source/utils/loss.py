import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-logit - max_val).exp()).log()
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss


class F2Loss(nn.Module):

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        probas = nn.Sigmoid()(output)
        probas = torch.clamp(probas * (1 - target), min=0.01) + probas * target
        TP = (probas * target).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (target.sum(dim=1) + self.epsilon)
        f1 = 5 * precision * recall / (4 * precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return -f1
