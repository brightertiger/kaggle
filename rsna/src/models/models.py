"""Original CNN families, with explicit control of pretrained downloads."""
import torch
import torch.nn as nn
import torchvision.models as models


class IntracranialHemorrhageModel(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

    def forward(self, x):
        raise NotImplementedError


class ResNetModel(IntracranialHemorrhageModel):
    def __init__(self, model_name='resnet50', num_classes=6, pretrained=True):
        super().__init__(num_classes, pretrained)
        if model_name not in ('resnet18', 'resnet50', 'resnet101'):
            raise ValueError(f'Unsupported ResNet model: {model_name}')
        self.backbone = getattr(models, model_name)(weights='DEFAULT' if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class InceptionModel(IntracranialHemorrhageModel):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__(num_classes, pretrained)
        # Pretrained torchvision weights require the auxiliary head during loading.
        self.backbone = models.inception_v3(weights='DEFAULT' if pretrained else None,
                                            aux_logits=pretrained, init_weights=False,
                                            transform_input=False)
        self.backbone.aux_logits = False
        self.backbone.AuxLogits = None
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ResNextModel(IntracranialHemorrhageModel):
    def __init__(self, model_name='se_resnext101_32x4d', num_classes=6, pretrained=True):
        super().__init__(num_classes, pretrained)
        if model_name not in ('se_resnext50_32x4d', 'se_resnext101_32x4d'):
            raise ValueError(f'Unsupported ResNext model: {model_name}')
        try:
            import pretrainedmodels
        except ImportError as exc:
            raise ImportError('SE-ResNeXt requires pretrainedmodels; install requirements.txt') from exc
        self.backbone = getattr(pretrainedmodels, model_name)(pretrained='imagenet' if pretrained else None)
        self.backbone.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.last_linear = nn.Linear(self.backbone.last_linear.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetModel(IntracranialHemorrhageModel):
    def __init__(self, model_name='efficientnet-b2', num_classes=6, pretrained=True):
        super().__init__(num_classes, pretrained)
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as exc:
            raise ImportError('EfficientNet requires efficientnet-pytorch; install requirements.txt') from exc
        factory = EfficientNet.from_pretrained if pretrained else EfficientNet.from_name
        self.backbone = factory(model_name, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


def create_model(model_name='resnext101', num_classes=6, pretrained=True):
    model_map = {
        'resnet18': lambda: ResNetModel('resnet18', num_classes, pretrained),
        'resnet50': lambda: ResNetModel('resnet50', num_classes, pretrained),
        'resnet101': lambda: ResNetModel('resnet101', num_classes, pretrained),
        'inception': lambda: InceptionModel(num_classes, pretrained),
        'resnext50': lambda: ResNextModel('se_resnext50_32x4d', num_classes, pretrained),
        'resnext101': lambda: ResNextModel('se_resnext101_32x4d', num_classes, pretrained),
        'efficientnet': lambda: EfficientNetModel('efficientnet-b2', num_classes, pretrained),
    }
    if model_name not in model_map:
        raise ValueError(f'Unknown model: {model_name}. Available models: {list(model_map)}')
    return model_map[model_name]()
