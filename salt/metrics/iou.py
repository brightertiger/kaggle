import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class IOUMetric(nn.Module):

    def __init__(self, cutoff, squash=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.squash = squash
        self.cutoff = cutoff
        return None
    
    def __eval__(self, actual, predict):
        if np.count_nonzero(actual) == 0 and np.count_nonzero(predict) > 0:
            return 0.
        elif np.count_nonzero(actual) >= 1 and np.count_nonzero(predict) == 0:
            return 0.
        elif np.count_nonzero(actual) == 0 and np.count_nonzero(predict) == 0:
            return 1.
        else:
            intersection = np.logical_and(actual, predict)
            union = np.logical_or(actual, predict)
            iou = np.sum(intersection > 0) / np.sum(union > 0)
            thresholds = np.arange(0.5, 1, 0.05)
            subscores = []
            for thresh in thresholds:
                subscores.append(iou > thresh)
            return np.mean(subscores)

    def forward(self, scores, actuals):
        actuals = actuals.squeeze(1).byte().cpu().data.numpy()
        if self.squash:
            scores = self.sigmoid(scores).squeeze(1)
        scores = (scores > self.cutoff).cpu().byte().data.numpy()
        batch_size = actuals.shape[0]
        metrics = []
        for batch in range(batch_size):
            actual = actuals[batch,:,:].astype(np.uint8)
            predict = scores[batch,:,:].astype(np.uint8)
            metrics.append(self.__eval__(actual, predict))
        return np.mean(metrics)

