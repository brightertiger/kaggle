import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models
torch.set_default_tensor_type('torch.DoubleTensor')

DROPOUT = 0.20

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        return None

    def forward(self, x):
        x = self.convolution(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        params = {}
        params['kernel_size'] = 3
        params['stride'] = 2
        params['padding'] = 1
        params['output_padding'] = 1
        self.convolution = EncoderBlock(in_channels, mid_channels)
        self.deconvolution = nn.ConvTranspose2d(mid_channels, out_channels, **params)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout2d(p=DROPOUT)
        return None

    def forward(self, x):
        x = self.convolution(x)
        x = self.deconvolution(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class UNetVGG11(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=DROPOUT)
        self.encoder = torchvision.models.vgg11_bn(pretrained=pretrained).features
        # Encoder
        self.encode1 = self.encoder[0]  
        self.encode2 = self.encoder[4]  
        self.encode3 = self.encoder[8]  
        self.encode4 = self.encoder[11] 
        self.encode5 = self.encoder[15]
        self.encode6 = self.encoder[18] 
        self.encode7 = self.encoder[22] 
        self.encode8 = self.encoder[25]
        # Batchnorm
        self.bchnrm1 = self.encoder[1]  
        self.bchnrm2 = self.encoder[5]  
        self.bchnrm3 = self.encoder[9]  
        self.bchnrm4 = self.encoder[12] 
        self.bchnrm5 = self.encoder[16]
        self.bchnrm6 = self.encoder[19] 
        self.bchnrm7 = self.encoder[23] 
        self.bchnrm8 = self.encoder[26]
        # Center
        self.center  = DecoderBlock(512, 512, 256)
        # Decoder
        self.decode5 = DecoderBlock(512 + 256, 512, 256)
        self.decode4 = DecoderBlock(512 + 256, 512, 128)
        self.decode3 = DecoderBlock(256 + 128, 256, 64)
        self.decode2 = DecoderBlock(64  + 128, 128, 32)
        self.decode1 = DecoderBlock(64  + 32, 64, 32)
        # Output
        self.predict = nn.Conv2d(32, 1, kernel_size=1)
        return None

    def forward(self, x):
        encode1 = self.encode1(x)
        encode1 = self.bchnrm1(encode1)
        encode1 = self.relu(encode1)
        pooled1 = self.pool(encode1)
        pooled1 = self.dropout(pooled1)
        encode2 = self.encode2(pooled1)
        encode2 = self.bchnrm2(encode2)
        encode2 = self.relu(encode2)
        pooled2 = self.pool(encode2)
        pooled2 = self.dropout(pooled2)
        encode3 = self.encode3(pooled2)
        encode3 = self.bchnrm3(encode3)
        encode3 = self.relu(encode3)
        encode4 = self.encode4(encode3)
        encode4 = self.bchnrm4(encode4)
        encode4 = self.relu(encode4)
        pooled4 = self.pool(encode4)
        pooled4 = self.dropout(pooled4)
        encode5 = self.encode5(pooled4)
        encode5 = self.bchnrm5(encode5)
        encode5 = self.relu(encode5)
        encode6 = self.encode6(encode5)
        encode6 = self.bchnrm6(encode6)
        encode6 = self.relu(encode6)
        pooled6 = self.pool(encode6)
        pooled6 = self.dropout(pooled6)
        encode7 = self.encode7(pooled6)
        encode7 = self.bchnrm7(encode7)
        encode7 = self.relu(encode7)
        encode8 = self.encode8(encode7)
        encode8 = self.bchnrm8(encode8)
        encode8 = self.relu(encode8)
        pooled8 = self.pool(encode8)
        pooled8 = self.dropout(pooled8)
        center  = self.center(pooled8)
        center  = self.pool(center)
        center = self.dropout(center)
        decode5 = self.decode5(torch.cat([center, pooled8],1))
        decode4 = self.decode4(torch.cat([decode5, pooled6],1))
        decode3 = self.decode3(torch.cat([decode4, pooled4],1))
        decode2 = self.decode2(torch.cat([decode3, pooled2],1))
        decode1 = self.decode1(torch.cat([decode2, pooled1],1))
        predict = self.predict(decode1)
        return predict