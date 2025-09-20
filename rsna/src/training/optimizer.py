import torch
import math
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, Optional

class RAdam(Optimizer):
    """
    RAdam optimizer implementation
    Paper: On the Variance of the Adaptive Learning Rate and Beyond
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for _ in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                p_data_fp32 = p.data.float()
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * 
                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)
                
                p.data.copy_(p_data_fp32)
        
        return loss

def create_optimizer(model: torch.nn.Module, optimizer_name: str = 'radam', 
                    lr: float = 1e-4, weight_decay: float = 1e-5, 
                    **kwargs) -> torch.optim.Optimizer:
    """Factory function to create optimizers"""
    
    optimizer_map = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'radam': RAdam,
    }
    
    if optimizer_name not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizer_map.keys())}")
    
    optimizer_class = optimizer_map[optimizer_name]
    
    if optimizer_name == 'sgd':
        return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)

def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str = 'step',
                    **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """Factory function to create learning rate schedulers"""
    
    scheduler_map = {
        'step': torch.optim.lr_scheduler.StepLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'multistep': torch.optim.lr_scheduler.MultiStepLR,
    }
    
    if scheduler_name not in scheduler_map:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {list(scheduler_map.keys())}")
    
    scheduler_class = scheduler_map[scheduler_name]
    
    if scheduler_name == 'step':
        return scheduler_class(optimizer, step_size=2, gamma=0.1)
    elif scheduler_name == 'cosine':
        return scheduler_class(optimizer, T_max=kwargs.get('epochs', 10))
    elif scheduler_name == 'plateau':
        return scheduler_class(optimizer, mode='min', patience=2, factor=0.5)
    elif scheduler_name == 'multistep':
        return scheduler_class(optimizer, milestones=[2, 4], gamma=0.1)
    else:
        return scheduler_class(optimizer, **kwargs)
