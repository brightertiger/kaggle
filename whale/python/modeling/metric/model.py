import torch
import sys
from torch import nn
from torchvision.models import resnet50

def load_model(model, path):
    results = torch.load(path)
    model.load_state_dict(results['model_state_dict'])
    loss = results['loss']
    print('Model Loaded:', 'Loss:', loss)
    sys.stdout.flush()
    return model

class Correct(nn.Module):

    def __init__(self):
        super().__init__()
        return None 
    
    def forward(self, output, target):
        batch_size = target.size(0)
        output = (output>0.5).float()
        correct = (output == target).float().sum()
        correct = correct.mul_(100.0 / batch_size)
        return correct

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

class CenterLoss(nn.Module):

    def __init__(self, num_classes=5004, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss

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
        self.head = nn.Sequential(*head)
        self.head.apply(self.__init_weights__)
        if freeze:
            for layer in list(self.backbone.children())[:-freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.classifier = nn.Linear(in_features=2048, out_features=5004, bias=True)
        self.embedding = nn.Linear(in_features=2048, out_features=256, bias=False)
        return None
    
    def __init_weights__(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
        return None
    
    def forward(self, image):
        feats = self.backbone(image)
        feats = self.head(feats)
        embed = self.embedding(feats)
        norm = embed.norm(p=2, dim=1, keepdim=True)
        embed = embed.div(norm.expand_as(embed))
        preds = self.classifier(feats)
        return preds, embed
    
class EmbedNet(nn.Module):

    def __init__(self, path, freeze=True, resnet=None):
        super(EmbedNet, self).__init__()
        self.model = load_model(ResNet(freeze=resnet), path)
        self.norm = nn.BatchNorm1d(1280)
        self.head = nn.Linear(1280, 1)
        self.sigmoid = nn.Sigmoid()
        self.head.apply(self.__init_weights__)
        if freeze:
            for layer in self.model.children():
                for param in layer.parameters():
                    param.requires_grad = False
        return None
    
    def __init_weights__(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
        return None
    
    def forward(self, image_1, image_2):
        _, embed_1 = self.model(image_1)
        _, embed_2 = self.model(image_2)
        add = embed_1 + embed_2 
        prd = embed_1 * embed_2 
        diff = torch.abs(embed_1 - embed_2)
        feats = torch.cat([embed_1, embed_2, add, prd, diff], dim=1)
        feats = self.norm(feats)
        output = self.sigmoid(self.head(feats)) 
        return output