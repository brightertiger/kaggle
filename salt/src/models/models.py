"""U-Net reconstruction: ResNet34/VGG11 encoders and optional decoder SCSE."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, vgg11, ResNet34_Weights, VGG11_Weights


class SCSEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(channels, max(1, channels // reduction), 1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(max(1, channels // reduction), channels, 1),
                                     nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)


class ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, inputs, skip, outputs, attention):
        super().__init__()
        self.conv = ConvBlock(inputs + skip, outputs)
        self.attention = SCSEModule(outputs) if attention else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.attention(self.conv(torch.cat([x, skip], dim=1)))


class SaltUNet(nn.Module):
    def __init__(self, model_name, config, pretrained):
        super().__init__()
        if config.TINY_MODEL:
            # The same skip-connected decoder with a narrow, random encoder for smoke tests.
            channels = [8, 16, 32, 64, 128]
            inputs = [3] + channels[:-1]
            self.encoder = nn.ModuleList([
                nn.Sequential(nn.AvgPool2d(2), ConvBlock(i, o))
                for i, o in zip(inputs, channels)
            ])
        elif model_name in {'resnet34', 'seresnet34'}:
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.ModuleList([
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
                nn.Sequential(backbone.maxpool, backbone.layer1),
                backbone.layer2, backbone.layer3, backbone.layer4,
            ])
            channels = [64, 64, 128, 256, 512]
        else:
            backbone = vgg11(weights=VGG11_Weights.DEFAULT if pretrained else None)
            stages, stage = [], []
            for layer in backbone.features:
                stage.append(layer)
                if isinstance(layer, nn.MaxPool2d):
                    stages.append(nn.Sequential(*stage))
                    stage = []
            self.encoder = nn.ModuleList(stages)
            channels = [64, 128, 256, 512, 512]
        attention = model_name == 'seresnet34'
        self.dropout = nn.Dropout2d(config.DROPOUT)
        self.decoder = nn.ModuleList([
            DecoderBlock(channels[i], channels[i - 1], channels[i - 1], attention)
            for i in range(4, 0, -1)
        ])
        self.head = nn.Sequential(ConvBlock(channels[0], channels[0]), nn.Conv2d(channels[0], 1, 1))

    def forward(self, images):
        features, x = [], images
        for stage in self.encoder:
            x = stage(x)
            features.append(x)
        x = self.dropout(x)
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            x = decoder(x, skip)
        x = F.interpolate(x, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)  # Raw logits for Lovasz hinge and threshold search.


def create_model(model_name, config, pretrained=None):
    if model_name not in {'resnet34', 'seresnet34', 'vgg11'}:
        raise ValueError(f'Unknown model: {model_name}')
    if pretrained is None:
        pretrained = config.PRETRAINED
    return SaltUNet(model_name, config, pretrained=pretrained and not config.TINY_MODEL)
