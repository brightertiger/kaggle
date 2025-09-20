import torch
import torch.nn as nn
from typing import Dict, Any
import torchvision.models as models
from torchvision.models import inception_v3

class IntracranialHemorrhageModel(nn.Module):
    """Base model class for intracranial hemorrhage detection"""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ResNetModel(IntracranialHemorrhageModel):
    """ResNet-based model for intracranial hemorrhage detection"""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 6, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class InceptionModel(IntracranialHemorrhageModel):
    """Inception-based model for intracranial hemorrhage detection"""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        self.backbone = inception_v3(pretrained=pretrained, aux_logits=False)
        self.backbone.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ResNextModel(IntracranialHemorrhageModel):
    """ResNext-based model for intracranial hemorrhage detection"""
    
    def __init__(self, model_name: str = 'se_resnext101_32x4d', num_classes: int = 6):
        super().__init__(num_classes, True)
        
        try:
            from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
            
            if model_name == 'se_resnext50_32x4d':
                self.backbone = se_resnext50_32x4d(pretrained='imagenet')
            elif model_name == 'se_resnext101_32x4d':
                self.backbone = se_resnext101_32x4d(pretrained='imagenet')
            else:
                raise ValueError(f"Unsupported ResNext model: {model_name}")
            
            self.backbone.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.last_linear = nn.Linear(2048, num_classes)
            
        except ImportError:
            print("Warning: pretrainedmodels not available, falling back to torchvision ResNet")
            self.backbone = models.resnet101(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class EfficientNetModel(IntracranialHemorrhageModel):
    """EfficientNet-based model for intracranial hemorrhage detection"""
    
    def __init__(self, model_name: str = 'efficientnet-b2', num_classes: int = 6):
        super().__init__(num_classes, True)
        
        try:
            import efficientnet_pytorch
            
            if hasattr(efficientnet_pytorch, model_name.replace('-', '_')):
                self.backbone = getattr(efficientnet_pytorch, model_name.replace('-', '_'))(
                    pretrained=True, num_classes=num_classes
                )
            else:
                print(f"Warning: {model_name} not available, using efficientnet-b0")
                self.backbone = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
                
        except ImportError:
            print("Warning: efficientnet_pytorch not available, falling back to ResNet")
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def create_model(model_name: str = 'resnext101', num_classes: int = 6) -> IntracranialHemorrhageModel:
    """Factory function to create models"""
    
    model_map = {
        'resnet50': lambda: ResNetModel('resnet50', num_classes),
        'resnet101': lambda: ResNetModel('resnet101', num_classes),
        'inception': lambda: InceptionModel(num_classes),
        'resnext50': lambda: ResNextModel('se_resnext50_32x4d', num_classes),
        'resnext101': lambda: ResNextModel('se_resnext101_32x4d', num_classes),
        'efficientnet': lambda: EfficientNetModel('efficientnet-b2', num_classes),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    return model_map[model_name]()
