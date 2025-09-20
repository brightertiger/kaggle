import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d

from .config import Config


class ResNextClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        if model_name == 'resnext50':
            self.backbone = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
            self.feature_dim = 2048
        elif model_name == 'resnext101':
            self.backbone = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self._setup_classifier()
        self._setup_freeze()
    
    def _setup_classifier(self):
        self.backbone.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Dropout(0.3)
        )
        self.backbone.last_linear = nn.Linear(self.feature_dim, self.num_classes, bias=True)
        self._init_weights(self.backbone.last_linear)
    
    def _setup_freeze(self):
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'last_linear' not in name:
                    param.requires_grad = False
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        targets = targets.float()
        
        max_val = (-logits).clamp(min=0)
        loss = logits - logits * targets + max_val + \
               ((-max_val).exp() + (-logits - max_val).exp()).log()
        
        invprobs = F.logsigmoid(-logits * (targets * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        
        if self.alpha is not None:
            loss = loss * self.alpha
        
        return loss.mean()


class F2Loss(nn.Module):
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        probas = torch.clamp(probas * (1 - targets), min=0.01) + probas * targets
        
        TP = (probas * targets).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (targets.sum(dim=1) + self.epsilon)
        
        f1 = 5 * precision * recall / (4 * precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        return -f1.mean()


class ModelFactory:
    @staticmethod
    def create_model(config: Config) -> ResNextClassifier:
        return ResNextClassifier(
            model_name=config.model_name,
            num_classes=config.num_classes,
            freeze_backbone=config.freeze_backbone
        )
    
    @staticmethod
    def create_loss_function(loss_type: str, config: Config) -> nn.Module:
        if loss_type == 'focal':
            return FocalLoss(gamma=config.focal_gamma)
        elif loss_type == 'f2':
            return F2Loss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


def load_model_checkpoint(model: ResNextClassifier, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metric': checkpoint.get('metric', 0.0)
    }


def save_model_checkpoint(model: ResNextClassifier, 
                         epoch: int, 
                         loss: float, 
                         metric: float, 
                         checkpoint_path: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'metric': metric
    }
    torch.save(checkpoint, checkpoint_path)
