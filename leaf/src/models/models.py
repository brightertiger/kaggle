import timm
import torch.nn as nn
from ..utils.config import Config

class EfficientNetModel(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

class CassavaClassifier(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
