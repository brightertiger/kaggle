import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing import Tuple, Optional

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size: Optional[Tuple[int, int]] = None):
        super().__init__()
        size = size or (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(size)
        self.maxpool = nn.AdaptiveMaxPool2d(size)
    
    def forward(self, x):
        return torch.cat([self.maxpool(x), self.avgpool(x)], 1)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)

class CenterLoss(nn.Module):
    def __init__(self, num_classes: int = 5004, feat_dim: int = 256, 
                 use_gpu: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Compute pairwise distances
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        
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

class WhaleResNet(nn.Module):
    def __init__(self, num_classes: int = 5004, freeze_layers: Optional[int] = None):
        super().__init__()
        
        # Load pretrained ResNet50 backbone
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Custom head
        head = [
            AdaptiveConcatPool2d(1),
            Flatten(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.25),
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.33)
        ]
        self.head = nn.Sequential(*head)
        self.head.apply(self._init_weights)
        
        # Classification and embedding layers
        self.classifier = nn.Linear(2048, num_classes, bias=True)
        self.embedding = nn.Linear(2048, 256, bias=False)
        
        # Freeze layers if specified
        if freeze_layers:
            for layer in list(self.backbone.children())[:-freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
    
    def forward(self, x):
        # Extract features
        feats = self.backbone(x)
        feats = self.head(feats)
        
        # Generate embeddings and predictions
        embed = self.embedding(feats)
        embed = F.normalize(embed, p=2, dim=1)  # L2 normalize embeddings
        
        preds = self.classifier(feats)
        return preds, embed

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_path: str, freeze_backbone: bool = True, 
                 resnet_layers: Optional[int] = None):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = WhaleResNet(freeze_layers=resnet_layers)
        if backbone_path:
            checkpoint = torch.load(backbone_path)
            self.backbone.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Siamese head
        self.norm = nn.BatchNorm1d(1280)  # 256 * 5 features
        self.head = nn.Linear(1280, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.head.apply(self._init_weights)
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
    
    def forward(self, image1, image2):
        # Get embeddings from both images
        _, embed1 = self.backbone(image1)
        _, embed2 = self.backbone(image2)
        
        # Create feature combinations
        add_feat = embed1 + embed2
        mul_feat = embed1 * embed2
        diff_feat = torch.abs(embed1 - embed2)
        
        # Concatenate all features
        combined_feats = torch.cat([embed1, embed2, add_feat, mul_feat, diff_feat], dim=1)
        
        # Apply normalization and classification
        combined_feats = self.norm(combined_feats)
        output = self.sigmoid(self.head(combined_feats))
        
        return output

class Accuracy(nn.Module):
    def __init__(self, topk: int = 5):
        super().__init__()
        self.topk = topk
    
    def forward(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0)
        result = correct_k.mul_(100.0 / batch_size)
        return result

class BinaryAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        batch_size = target.size(0)
        output = (output > 0.5).float()
        correct = (output == target).float().sum()
        correct = correct.mul_(100.0 / batch_size)
        return correct
