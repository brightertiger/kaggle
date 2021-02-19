import timm
import math
import torch
from torch import nn
from torch.nn import functional as F

class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, train, label=False):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if label:
            one_hot = torch.zeros(cosine.size(), device='cuda:0')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        else:
            output = cosine
        output *= self.s
        return output

class GeM(nn.Module):
    
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        

class EfficientNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.base = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        self.gem = GeM()
        n_features = self.base.classifier.in_features
        self.fc = nn.Linear(n_features, 1024)
        self.arcface = ArcMarginProduct(1024, 5)
        return None
    
    def forward(self, x, label=False):
        x = self.base.forward_features(x)
        x = self.gem(x).squeeze()
        x = self.fc(x)
        if self.training:
            x = self.arcface(x, True, label)
        else:
            x = self.arcface(x, False)
        return x