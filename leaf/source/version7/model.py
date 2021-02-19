import timm
import torch
from torch import nn

    
class EfficientModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 5)
        return None

    def forward(self, image):
        return self.model(image)