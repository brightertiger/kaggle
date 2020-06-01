import torch
from torch import nn
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
from torchvision.models import inception_v3
    
class InceptionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Linear(2048, 6, bias=True)
        return None

    def forward(self, image):
        output = self.model(image)
        return output
    
class ResNextModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = se_resnext101_32x4d(pretrained='imagenet')
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(2048, 6)  
        return None

    def forward(self, image):
        output = self.model(image)
        return output   