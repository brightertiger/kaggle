import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import res2next50

class ResnetModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = res2next50(pretrained=True)
        self.model.fc = nn.Linear(2048, 24)
        return None
    
    def forward(self, image):
        output = self.model(image)
        return output