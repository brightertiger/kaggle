import torch
from torch import nn
from torchvision.models import resnet50

class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(2048, 340)
        return None

    def forward(self, x):
        x = x.expand(-1,3,-1,-1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x