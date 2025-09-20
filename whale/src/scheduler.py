import torch
import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warm restarts.
    
    This scheduler implements the cosine annealing with warm restarts as described in
    "SGDR: Stochastic Gradient Descent with Warm Restarts" by Loshchilov & Hutter.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        T_mult (float): Factor to increase the period after each restart. Default: 1.
    """
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        super().__init__(optimizer, last_epoch)

    def restart(self):
        """Restart the scheduler with increased period."""
        self.restart_every = int(self.restart_every * self.T_mult)
        self.restarted_at = self.last_epoch
        # Reduce learning rate by 10% after each restart
        self.base_lrs = [0.9 * base_lr for base_lr in self.base_lrs]

    def cosine(self, base_lr):
        """Compute cosine annealing learning rate."""
        return np.max([
            self.eta_min, 
            base_lr * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2
        ])

    @property
    def step_n(self):
        """Number of steps since last restart."""
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        """Get current learning rate."""
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]
