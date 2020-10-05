import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
    
class EfficientModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b5', advprop=True)
        self.net._fc = nn.Linear(2048, 1, bias=True)

    def forward(self, image):
        return self.net(image)
    
    