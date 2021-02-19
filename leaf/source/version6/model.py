import timm
import torch
from torch import nn

class ViTBase16(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_384", pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 5)

    def forward(self, image):
        output = self.model(image)
        return output