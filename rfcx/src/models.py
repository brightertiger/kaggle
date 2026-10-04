import timm
import torch
import torch.nn as nn
from .config import Config

class AudioClassificationModel(nn.Module):
    def __init__(self, config: Config, model_name: str = "res2net50_26w_4s"):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        self.backbone = timm.create_model(
            config.model.backbone or model_name,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
        )
    
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
