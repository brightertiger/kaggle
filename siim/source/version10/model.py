import torch
from torch import nn

PATH = "facebookresearch/semi-supervised-ImageNet1K-models"


class ResNestModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        self.net.fc = nn.Linear(2048,4, bias=True)

    def forward(self, image):
        return self.net(image)
    
    