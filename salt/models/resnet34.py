import torch
import torchvision
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import models
torch.set_default_tensor_type('torch.DoubleTensor')

DROPOUT = 0.25

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=DROPOUT)
        return None

    def forward(self, x):
        x = self.convolution(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, is_deconv=True):
        super().__init__()
        self.dropout = nn.Dropout2d(p=DROPOUT)
        block = []
        block += [EncoderBlock(in_channels, mid_channels)]
        block += [nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        block += [nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*block)
        return None

    def forward(self, x):
        x = self.block(x)
        x = self.dropout(x)
        return x

class UNetResNet34(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.dropout = nn.Dropout2d(p=DROPOUT)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        # Encoder
        inputs = [self.encoder.conv1,self.encoder.bn1,self.encoder.relu]
        self.inputs  = nn.Sequential(*inputs)
        self.encode1 = self.encoder.layer1
        self.encode2 = self.encoder.layer2
        self.encode3 = self.encoder.layer3
        self.center  = self.encoder.layer4
        # Decoder
        self.decode4 = DecoderBlock(512, 256, 128)
        self.decode3 = DecoderBlock(256 + 128, 256, 128)
        self.decode2 = DecoderBlock(128 + 128, 128, 32)
        self.decode1 = DecoderBlock(64 + 32, 64, 64)
        # Output
        self.predict = nn.Conv2d(64, 1, kernel_size=1)
        return None

    def forward(self, x):
        inputs  = self.inputs(x)
        encode1 = self.encode1(inputs)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        center  = self.center(encode3)
        decode4 = self.decode4(center)
        decode3 = self.decode3(torch.cat([decode4,encode3],1))
        decode2 = self.decode2(torch.cat([decode3,encode2],1))
        decode1 = self.decode1(torch.cat([decode2,encode1],1))
        predict = self.predict(decode1)
        return predict