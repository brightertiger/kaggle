import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
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
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        if use_gpu and torch.cuda.is_available():
            self.cuda()

    def forward(self, x, labels):
        # Same squared distance objective, without allocating batch x all classes.
        return (x - self.centers[labels]).square().sum(dim=1).clamp(1e-12, 1e12).mean()

class WhaleResNet(nn.Module):
    def __init__(self, num_classes: int = 5004, freeze_layers: Optional[int] = None,
                 embedding_dim: int = 256, pretrained: bool = True,
                 backbone_name: str = "resnet50", head_dim: int = 2048):
        super().__init__()
        
        # Load pretrained ResNet50 backbone
        factories = {"resnet50": (resnet50, ResNet50_Weights.DEFAULT),
                     "resnet18": (resnet18, ResNet18_Weights.DEFAULT)}
        if backbone_name not in factories:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        factory, weights = factories[backbone_name]
        backbone = factory(weights=weights if pretrained else None)
        pooled_dim = backbone.fc.in_features * 2
        self.model_kwargs = dict(num_classes=num_classes, embedding_dim=embedding_dim,
                                 backbone_name=backbone_name, head_dim=head_dim)
        self.class_names = None
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Custom head
        head = [
            AdaptiveConcatPool2d(1),
            Flatten(),
            nn.BatchNorm1d(pooled_dim),
            nn.Dropout(0.25),
            nn.Linear(pooled_dim, head_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(head_dim),
            nn.Dropout(0.33)
        ]
        self.head = nn.Sequential(*head)
        self.head.apply(self._init_weights)
        
        # Classification and embedding layers
        self.classifier = nn.Linear(head_dim, num_classes, bias=True)
        self.embedding = nn.Linear(head_dim, embedding_dim, bias=False)
        
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
    def __init__(self, backbone_path: Optional[str] = None, freeze_backbone: bool = True,
                 resnet_layers: Optional[int] = None, model_kwargs=None):
        super().__init__()
        
        checkpoint = None
        if backbone_path:
            checkpoint = torch.load(backbone_path, map_location="cpu", weights_only=True)
            model_kwargs = checkpoint.get('model_kwargs', model_kwargs)
        model_kwargs = dict(model_kwargs or {})
        model_kwargs['pretrained'] = False
        self.backbone = WhaleResNet(freeze_layers=resnet_layers, **model_kwargs)
        if checkpoint:
            self.backbone.load_state_dict(checkpoint['model_state_dict'])
        self.model_kwargs = self.backbone.model_kwargs
        self.class_names = checkpoint.get('class_names') if checkpoint else None
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        feature_dim = self.model_kwargs['embedding_dim'] * 5
        self.norm = nn.BatchNorm1d(feature_dim)
        self.head = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.head.apply(self._init_weights)
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # Keep frozen BatchNorm buffers and dropout fixed.
        return self

    def forward(self, image1, image2):
        # Get embeddings from both images
        _, embed1 = self.backbone(image1)
        _, embed2 = self.backbone(image2)
        
        return self.score_embeddings(embed1, embed2)

    def score_embeddings(self, embed1, embed2):
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
        _, pred = output.topk(min(self.topk, output.shape[1]), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].reshape(-1).float().sum(0)
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


class MeanAveragePrecision(nn.Module):
    """MAP@k for one relevant identity per image (mean reciprocal rank)."""
    def __init__(self, topk=5):
        super().__init__()
        self.topk = topk

    def forward(self, output, target):
        k = min(self.topk, output.shape[1])
        indices = output.topk(k, dim=1).indices
        hits = indices.eq(target.reshape(-1, 1))
        ranks = torch.arange(1, k + 1, device=output.device)
        return (hits / ranks).sum(dim=1).mean()
