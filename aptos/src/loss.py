import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        target = torch.zeros_like(logits)
        target = target.scatter(1, labels.reshape(-1, 1), 1)
        target = target.float()
        target = target * 0.9 + 0.02
        
        loss = self.bce_loss(logits, target)
        loss = loss.sum(dim=-1)
        
        return loss

class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, output_1: torch.Tensor, output_2: torch.Tensor) -> torch.Tensor:
        loss = torch.pow(output_1 - output_2, 2)
        loss = loss.sum(dim=-1)
        return loss

class DiabeticRetinopathyLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.75, variance_weight: float = 0.0, config: Config = None):
        super().__init__()
        self.config = config or Config()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = FocalBCELoss()
        self.variance_loss = VarianceLoss()
        self.mse_weight = mse_weight
        self.variance_weight = variance_weight
    
    def forward(self, regression_output: torch.Tensor, classification_output: torch.Tensor, 
                labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        regression_output = torch.clamp(regression_output, -0.5, 4.5)
        
        mse_loss = self.mse_loss(regression_output, labels)
        bce_loss = self.bce_loss(classification_output, labels.long())
        
        loss = (self.mse_weight * mse_loss) + ((1.0 - self.mse_weight) * bce_loss)
        loss = loss * weights
        
        return loss

class NoiseAugmentedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.75, variance_weight: float = 0.2, config: Config = None):
        super().__init__()
        self.config = config or Config()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = FocalBCELoss()
        self.variance_loss = VarianceLoss()
        self.mse_weight = mse_weight
        self.variance_weight = variance_weight
    
    def forward(self, regression_1: torch.Tensor, regression_2: torch.Tensor,
                classification_1: torch.Tensor, classification_2: torch.Tensor,
                labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        regression_output = (regression_1 + regression_2) / 2
        classification_output = (classification_1 + classification_2) / 2
        
        regression_output = torch.clamp(regression_output, -0.5, 4.5)
        
        mse_loss = self.mse_loss(regression_output, labels)
        bce_loss = self.bce_loss(classification_output, labels.long())
        variance_loss = (self.variance_loss(regression_1, regression_2) + 
                        self.variance_loss(classification_1, classification_2))
        
        loss = (self.mse_weight * mse_loss) + ((1.0 - self.mse_weight) * bce_loss) + (self.variance_weight * variance_loss)
        loss = loss * weights
        
        return loss
