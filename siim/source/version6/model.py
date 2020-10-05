import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
    
class EfficientModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b4', advprop=True)
        self.net._fc = nn.Linear(1792, 4, bias=True)

    def forward(self, image):
        return self.net(image)
    
    