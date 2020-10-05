import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
    
class EfficientModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b5', advprop=True)
        self.net._fc = nn.Linear(2048, 512, bias=True)
        self.metad = nn.Linear(13, 256, bias=True)
        self.out = nn.Sequential(nn.Linear(512 + 256, 256), nn.ReLU(), nn.Linear(256, 4))
        return None

    def forward(self, image, metad):
        image = self.net(image)
        metad = self.metad(metad)
        feats = torch.cat([image, metad], dim=-1)
        out = self.out(feats)
        return out
    
    