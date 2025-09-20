import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from config import Config

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int = 2048, num_classes: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 1, bias=True)
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, num_classes, bias=True)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in [self.regression_head, self.classification_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
    
    def forward(self, features: torch.Tensor) -> tuple:
        regression_output = self.regression_head(features)
        classification_output = self.classification_head(features)
        return regression_output.squeeze(), classification_output

class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, model_name: str = "efficientnet-b5", config: Config = None):
        super().__init__()
        self.config = config or Config()
        self.backbone = EfficientNet.from_pretrained(model_name)
        self.backbone._fc = ClassificationHead(
            input_dim=2048,
            num_classes=5,
            dropout_rate=self.config.DROPOUT_RATE
        )
    
    def forward(self, image: torch.Tensor) -> tuple:
        regression_output, classification_output = self.backbone(image)
        return regression_output, classification_output
