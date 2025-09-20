import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for intracranial hemorrhage detection"""
    
    def __init__(self, class_weights: Tuple[float, ...] = None):
        super().__init__()
        
        if class_weights is None:
            self.class_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            self.class_weights = torch.tensor(class_weights)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
        Returns:
            Weighted BCE loss
        """
        weights = self.class_weights.to(logits.device)
        
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            weight=weights, 
            reduction='none'
        )
        
        return loss

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
        Returns:
            Focal loss
        """
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation-like tasks"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
        Returns:
            Dice loss
        """
        probs = torch.sigmoid(logits)
        
        intersection = (probs * targets).sum(dim=0)
        union = probs.sum(dim=0) + targets.sum(dim=0)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class CombinedLoss(nn.Module):
    """Combined loss function"""
    
    def __init__(self, bce_weight: float = 1.0, focal_weight: float = 0.0, dice_weight: float = 0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = WeightedBCELoss()
        if focal_weight > 0:
            self.focal_loss = FocalLoss()
        if dice_weight > 0:
            self.dice_loss = DiceLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
        Returns:
            Combined loss
        """
        total_loss = self.bce_weight * self.bce_loss(logits, targets).mean()
        
        if self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal_loss(logits, targets).mean()
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(logits, targets)
        
        return total_loss
