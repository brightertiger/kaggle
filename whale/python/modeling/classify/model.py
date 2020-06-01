import torch
from torch import nn
from torchvision.models import resnet50

class Accuracy(nn.Module):
    def __init__(self, topk=5):
        super().__init__()
        self.topk = topk 
        return None

    def forward(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))        
        correct_k = correct[:self.topk].view(-1).float().sum(0)
        result = (correct_k.mul_(100.0 / batch_size))
        return result

class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, size=None):
        super().__init__()
        size = size or (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d(size)
        self.maxpool = nn.AdaptiveMaxPool2d(size)
        return None

    def forward(self, x):
        return torch.cat([self.maxpool(x), self.avgpool(x)], 1)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()
        return None

    def forward(self, x):
        return x.view(x.size()[0], -1)

class ResNet(nn.Module):

    def __init__(self, freeze=None):
        super(ResNet, self).__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        head = []
        head += [AdaptiveConcatPool2d(1)]
        head += [Flatten()]
        head += [nn.BatchNorm1d(4096)]
        head += [nn.Dropout(0.25)]
        head += [nn.Linear(in_features=4096, out_features=2048, bias=False)]
        head += [nn.ReLU()]
        head += [nn.BatchNorm1d(2048)]
        head += [nn.Dropout(0.33)]
        head += [nn.Linear(in_features=2048, out_features=5004, bias=True)]
        self.head = nn.Sequential(*head)
        self.head.apply(self.__init_weights__)
        if freeze:
            for layer in list(self.backbone.children())[:-freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
        return None
    
    def __init_weights__(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
        return None
    
    def forward(self, x):
        feats = self.backbone(x)
        output = self.head(feats)
        return output
