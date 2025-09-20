import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetClassifier(nn.Module):
    def __init__(self, 
                 model_name: str = 'resnet50',
                 num_classes: int = 340,
                 pretrained: bool = True):
        super(ResNetClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            fc_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            fc_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            fc_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(fc_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TopKAccuracy(nn.Module):
    def __init__(self, k: int = 3):
        super(TopKAccuracy, self).__init__()
        self.k = k

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = target.size(0)
        _, pred = output.topk(self.k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.k].view(-1).float().sum(0)
        result = correct_k.mul_(100.0 / batch_size)
        return result
