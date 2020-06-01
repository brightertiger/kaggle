import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
    
class Features(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.regress = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 1, bias=True))
        self.classify = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 5, bias=True))
        self.__init_weights__(self.regress)
        self.__init_weights__(self.classify)
        return None
    
    def __init_weights__(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
        return None
    
    def forward(self, features):
        regress = self.regress(features)
        classify = self.classify(features)
        return regress.squeeze(), classify

class EfficientModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.model._fc = Features()
        return None

    def forward(self, image):
        regress, classify = self.model(image)
        return regress, classify
            