"""Binary losses and the competition's per-image IoU threshold score.

Lovasz hinge follows the sorted-error formulation in Berman et al.:
https://arxiv.org/abs/1705.08790
"""
import torch
from torch import nn
from torch.nn import functional as F


class LovaszLoss(nn.Module):
    def forward(self, logits, targets):
        losses = []
        for prediction, target in zip(logits, targets):
            target = target.flatten().to(prediction.dtype)
            margins = 1 - prediction.flatten() * (2 * target - 1)
            errors, permutation = margins.sort(descending=True)
            foreground = target[permutation]
            remaining = foreground.sum() - foreground.cumsum(0)
            union = foreground.sum() + (1 - foreground).cumsum(0)
            jaccard = 1 - remaining / union.clamp_min(1)
            weights = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
            losses.append((errors.relu() * weights).sum())
        return torch.stack(losses).mean()


class DiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight, self.bce_weight = dice_weight, bce_weight

    def forward(self, logits, targets):
        probabilities, truth = logits.sigmoid().flatten(1), targets.flatten(1)
        dice = (2 * (probabilities * truth).sum(1) + 1) / (probabilities.sum(1) + truth.sum(1) + 1)
        return self.dice_weight * (1 - dice.mean()) + self.bce_weight * F.binary_cross_entropy_with_logits(logits, targets)


class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return ((1 - torch.exp(-bce)) ** 2 * bce).mean()


class IOUMetric:
    """Mean precision over IoU thresholds 0.50..0.95; both empty scores one."""
    def __init__(self, cutoff=-0.18, squash=False, min_salt_pixels=0):
        self.cutoff, self.squash = cutoff, squash
        self.min_salt_pixels = min_salt_pixels

    def __call__(self, scores, targets):
        if self.squash:
            scores = scores.sigmoid()
        predicted = (scores >= self.cutoff).flatten(1)
        truth = (targets > 0.5).flatten(1)
        predicted = predicted & (predicted.sum(1, keepdim=True) > self.min_salt_pixels)
        intersection = (predicted & truth).sum(1).double()
        union = (predicted | truth).sum(1).double()
        iou = torch.where(union == 0, torch.ones_like(union), intersection / union.clamp_min(1))
        thresholds = torch.tensor([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                                  device=iou.device, dtype=iou.dtype)
        return (iou[:, None] > thresholds).double().mean().item()


def create_loss_function(name, options=None):
    options = options or {}
    if name == 'lovasz':
        return LovaszLoss()
    if name == 'dice':
        return DiceLoss(**options)
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    if name == 'focal':
        return FocalLoss()
    raise ValueError(f'Unknown loss: {name}')


def create_metric_function(name, options=None):
    if name != 'iou':
        raise ValueError(f'Unknown metric: {name}')
    return IOUMetric(**(options or {}))
