import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .config import Config

class AudioClassificationModel(nn.Module):
    def __init__(self, config: Config, model_name: str = "res2net50_26w_4s"):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        if model_name == "res2net50_26w_4s":
            self.backbone = timm.create_model("res2net50_26w_4s", pretrained=config.model.pretrained)
            self.backbone.fc = nn.Linear(2048, config.model.num_classes)
        elif model_name == "resnest50d":
            self.backbone = timm.create_model("resnest50d", pretrained=config.model.pretrained)
            self.backbone.fc = nn.Linear(2048, config.model.num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ResNetModel(AudioClassificationModel):
    def __init__(self, config: Config):
        super().__init__(config, "res2net50_26w_4s")

class ResNeStModel(AudioClassificationModel):
    def __init__(self, config: Config):
        super().__init__(config, "resnest50d")

def create_model(config: Config, model_type: str = "resnet") -> nn.Module:
    if model_type.lower() == "resnet":
        return ResNetModel(config)
    elif model_type.lower() == "resnest":
        return ResNeStModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
