import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
torch.set_default_tensor_type('torch.DoubleTensor')

DROPOUT = 0.25

class SCSEModule(nn.Module):

    def __init__(self, ch, re=16):
        super().__init__()
        cSE  = [nn.AdaptiveAvgPool2d(1),nn.Conv2d(ch,ch//re,1)]
        cSE += [nn.ReLU(inplace=True),nn.Conv2d(ch//re,ch,1),nn.Sigmoid()]
        self.cSE = nn.Sequential(*cSE)
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),nn.Sigmoid())
        return None

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class ConvBlock(nn.Module):

    def __init__(self, input, output, kernel=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(output)
        return None

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.dropout2d(x, p=0.3)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upsample=True):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SCSEModule(out_channels)
        self.upsample = upsample

    def forward(self, x, y=None):
        if self.upsample:
            x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if y is not None:
            x = torch.cat([x,y], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.dropout2d(x, p=0.3)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x
 
class UNetResNet34(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.dropout = nn.Dropout2d(p=DROPOUT)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        # Encoder
        inputs = [self.encoder.conv1,self.encoder.bn1,self.encoder.relu]
        self.inputs  = nn.Sequential(*inputs)
        self.encode1 = self.encoder.layer1 # 64
        self.encode2 = self.encoder.layer2 # 128
        self.encode3 = self.encoder.layer3 # 256
        self.encode4 = self.encoder.layer4 # 512
        # Center
        center = [ConvBlock(512, 512, kernel=3, padding=1), nn.ReLU(inplace=True)]
        center += [ConvBlock(512, 256, kernel=3, padding=1), nn.ReLU(inplace=True)]
        self.center = nn.Sequential(*center)
        # Decoder
        self.decode5 = DecoderBlock(512 + 256, 512, 256)
        self.decode4 = DecoderBlock(256 + 256, 512, 256)
        self.decode3 = DecoderBlock(128 + 256, 256,  64)
        self.decode2 = DecoderBlock(64 + 64  , 128, 128)
        self.decode1 = DecoderBlock(64 + 64  , 128,  32, False)
        # Output
        predict  = [nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        predict += [nn.Conv2d(32,  1, kernel_size=1, padding=0)]
        self.predict = nn.Sequential(*predict)
        return None

    def forward(self, x):
        inputs  = self.inputs(x)
        encode1 = self.encode1(inputs)
        encode1 = F.dropout2d(encode1, p=0.3)
        encode2 = self.encode2(encode1)
        encode2 = F.dropout2d(encode2, p=0.3)
        encode3 = self.encode3(encode2)
        encode3 = F.dropout2d(encode3, p=0.3)
        encode4 = self.encode4(encode3)
        encode4 = F.dropout2d(encode4, p=0.3)
        center  = self.center(encode4)
        center  = F.dropout2d(center, p=0.3)
        decode5 = self.decode5(torch.cat([center,encode4],1))
        decode5 = F.dropout2d(decode5, p=0.3)
        decode4 = self.decode4(torch.cat([decode5,encode3],1))
        decode4 = F.dropout2d(decode4, p=0.3)
        decode3 = self.decode3(torch.cat([decode4,encode2],1))
        decode3 = F.dropout2d(decode3, p=0.3)
        decode2 = self.decode2(torch.cat([decode3,encode1],1))
        decode2 = F.dropout2d(decode2, p=0.3)
        decode1 = self.decode1(decode2)
        decode1 = F.dropout2d(decode1, p=0.3)
        predict = self.predict(decode1)
        return predict