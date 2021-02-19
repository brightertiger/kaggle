import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CELoss(nn.Module):
    
    def __init__(self, epsilon=5e-4):
        super().__init__()
        self.num_classes = 5
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        targets = F.one_hot(targets, num_classes=5)
        log_probs = self.logsoftmax(inputs)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
