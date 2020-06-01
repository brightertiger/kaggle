import torch 
from torch import nn
import numpy as np 

class Accuracy(nn.Module):

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        return None

    def forward(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))        
        correct_k = correct[:self.topk].view(-1).float().sum(0)
        result = (correct_k.mul_(100.0 / batch_size))
        return result