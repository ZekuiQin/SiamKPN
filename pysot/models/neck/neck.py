from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
from pysot.core.config import cfg

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return [self.downsample(features[-1])]
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

class Keypoint_Unsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Keypoint_Unsample, self).__init__()
        self.num = len(out_channels)
        assert self.num==5,'should input 5 convs'
        self.adjust_conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            )
        self.adjust_conv2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
            )
        self.adjust_conv3 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[2]),
            )
        self.adjust_conv4 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[3]),
            )
        self.adjust_conv5 = nn.Sequential(
            nn.Conv2d(in_channels[4], out_channels[4], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[4]),
            )

