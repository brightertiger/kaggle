import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CELoss(nn.Module):
    
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.num_classes = 5
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        return None

    def forward(self, inputs, targets):
        if torch.randperm(2)[1].data.item() > 0:
            batch_size = inputs.size(0) - 1
        else:
            batch_size = inputs.size(0)
        log_probs = self.logsoftmax(inputs)
        loss = (- targets * log_probs).sum(1)
        sort, idx = torch.sort(loss, descending=False)
        batch_size = idx[:batch_size]
        loss = loss[batch_size]
        loss = loss.mean(0).sum()
        return loss
