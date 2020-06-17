import torch
import torch.nn as nn
from models.spectral_norm import SpectralNorm

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation='LeakyReLU', norm=None, init_method=None, std=1., sn=False):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True  # 后边有batchnorm层，不需要设置bias,因为会将输出归一化，设置偏置没有用
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                bias=bias)
        if sn:
            self.conv2d = SpectralNorm(self.conv2d)  #谱归一化
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = getattr(torch.nn, activation, 'LeakyReLU') #从torch.nn中返回activation属性，如果不存在，则返回字符串'LeakyReLU'
                self.activation = self.activation()
            else:
                self.activation = getattr(torch, activation, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.01)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True) #每个sample,每个通道进行Norm

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None, sn=False):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        if sn:
            self.conv1 = SpectralNorm(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=bias))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        if sn:
            self.conv2 = SpectralNorm(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=bias))
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual  #卷积前矩阵与卷积后矩阵相加
        out = self.relu(out)
        return out
