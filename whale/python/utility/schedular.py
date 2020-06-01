import torch
import math 
import numpy as np

class CosineLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        super().__init__(optimizer, last_epoch)

    def restart(self):
        self.restart_every *= self.T_mult
        self.restarted_at = self.last_epoch
        self.base_lrs = [0.9 * base_lr for base_lr in self.base_lrs]

    def cosine(self, base_lr):
        return np.max([self.eta_min, base_lr * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2])

    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]