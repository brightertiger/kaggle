import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class LovaszLoss(nn.Module):

    def __init__(self):
        super().__init__() 
        return None

    def __gradient__(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def __eval__(self, logits, labels):
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.__gradient__(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss
    
    def mean(self, l, empty=0):
        l = iter(l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def forward(self, logits, labels):
        loss = []
        for logit, label in zip(logits, labels):
            logit = logit.unsqueeze(0).view(-1)
            label = label.unsqueeze(0).view(-1)
            loss.append(self.__eval__(logit, label))
        return self.mean(loss)


