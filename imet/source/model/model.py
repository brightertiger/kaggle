import torch
from torch import nn
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d

class ResNext50(nn.Module):

    def __init__(self, freeze):
        super().__init__()
        self.model = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        self.model.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Dropout(0.3))
        self.model.last_linear = nn.Linear(2048, 1103, bias=True)
        self.__init_weights__(self.model.last_linear)
        if freeze:
            for name, param in self.model.named_parameters():
                if 'last_linear' not in name:
                    param.requires_grad = False
        return None

    def __init_weights__(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
        return None

    def forward(self, image):
        output = self.model(image)
        return output
    
class ResNext100(nn.Module):

    def __init__(self, freeze):
        super().__init__()
        self.model = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        self.model.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Dropout(0.3))
        self.model.last_linear = nn.Linear(2048, 1103, bias=True)
        self.__init_weights__(self.model.last_linear)
        if freeze:
            for name, param in self.model.named_parameters():
                if 'last_linear' not in name:
                    param.requires_grad = False
        return None

    def __init_weights__(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
        return None

    def forward(self, image):
        output = self.model(image)
        return output