from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.core.xcorr import xcorr_fast, xcorr_depthwise

class KPN(nn.Module):
    def __init__(self):
        super(KPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, with_padding=True, groups=1):
        super(ConvBN,self).__init__()
        pad = (kernel_size-1)//2 if with_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)

        return bn


class StackXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(StackXCorr, self).__init__()
        self.z_conv1 = ConvBN(in_channels, hidden, kernel_size=3, with_padding=False)
        self.x_conv1 = ConvBN(in_channels, hidden, kernel_size=3, with_padding=True)
        if cfg.TRAIN.STACK==1 or cfg.TRAIN.INTER_SUPER:
            self.head1 = nn.Sequential(
                    ConvBN(hidden,hidden,kernel_size=1,with_padding=False),
                    nn.Conv2d(hidden, out_channels, kernel_size=1)
                    )

        if cfg.TRAIN.STACK>=2:
            self.z_conv2 = ConvBN(hidden, hidden, kernel_size=3, with_padding=True)
            self.z_conv22 = ConvBN(hidden, hidden, kernel_size=3, with_padding=False)
            self.x_conv2 = ConvBN(hidden, hidden, kernel_size=3, with_padding=True)
            self.head2 = nn.Sequential(
                    ConvBN(hidden,hidden,kernel_size=1,with_padding=False),
                    nn.Conv2d(hidden, out_channels, kernel_size=1)
                    )

        if cfg.TRAIN.STACK==3:
            self.z_conv3 = ConvBN(hidden, hidden, kernel_size=3, with_padding=True)
            self.z_conv32 = ConvBN(hidden, hidden, kernel_size=3, with_padding=False)
            self.x_conv3 = ConvBN(hidden, hidden, kernel_size=3, with_padding=True)
            self.head3 = nn.Sequential(
                    ConvBN(hidden,hidden,kernel_size=1,with_padding=False),
                    nn.Conv2d(hidden, out_channels, kernel_size=1)
                    )

    def forward(self, kernel, search):
        kernel1 = kernel[:, :, 4:11, 4:11]
        kernel1 = self.z_conv1(kernel1)
        search1 = self.x_conv1(search)

        search1 = F.pad(search1,(2,2,2,2))
        feature1 = xcorr_depthwise(search1, kernel1)

        out1 = self.head1(feature1)
        out = []
        out.append(out1)
        if cfg.TRAIN.STACK==1:
            return out

        kernel20 = self.z_conv2(kernel)
        kernel2 = kernel20[:, :, 4:11, 4:11]
        kernel2 = self.z_conv22(kernel2)
        search2 = self.x_conv2(feature1)
        search2 = F.pad(search2,(2,2,2,2))
        feature2 = xcorr_depthwise(search2, kernel2)

        out2 = self.head2(feature2)
        out.append(out2)
        if cfg.TRAIN.STACK==2:
            return out

        kernel3 = self.z_conv3(kernel20)
        kernel3 = kernel3[:,:,4:11,4:11]
        kernel3 = self.z_conv32(kernel3)
        search3 = self.x_conv3(feature2)
        search3 = F.pad(search3,(2,2,2,2))
        feature3 = xcorr_depthwise(search3, kernel3)
        out3 = self.head3(feature3)
        out.append(out3)
        return out
        

class DepthwiseKPN(KPN):
    def __init__(self, in_channels=256, out_channels=256):
        super(DepthwiseKPN, self).__init__()
        self.heatmap = StackXCorr(in_channels, out_channels, 1)
        if cfg.TRAIN.OFFSETS:
            self.offsets = StackXCorr(in_channels, out_channels, 2)
        self.objsize = StackXCorr(in_channels, out_channels, 2)

    def forward(self, z_f, x_f):
        if cfg.TRAIN.CROP_TEMPLATE:
            l = 8
            r = l +16
            z_f = z_f[:, :, l:r, l:r]
        heatmap = self.heatmap(z_f, x_f)
        objsize = self.objsize(z_f, x_f)
        if cfg.TRAIN.OFFSETS:
            offsets = self.offsets(z_f, x_f)
            return heatmap, offsets, objsize
        else:
            return heatmap, objsize


class MultiKPN(KPN):
    def __init__(self, in_channels, weighted=False):
        super(MultiKPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('kpn'+str(i),
                    DepthwiseKPN(in_channels[i], in_channels[i]))
        if self.weighted:
            self.heatmap_weight = nn.Parameter(torch.ones(len(in_channels)))
            if cfg.TRAIN.OFFSETS:
                self.offsets_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.objsize_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        heatmap = [[] for i in range(cfg.TRAIN.STACK)]
        offsets = [[] for i in range(cfg.TRAIN.STACK)]
        objsize = [[] for i in range(cfg.TRAIN.STACK)]

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=0):
            kpn = getattr(self, 'kpn'+str(idx))
            if cfg.TRAIN.OFFSETS:
                h, o, s = kpn(z_f, x_f)
                for i in range(cfg.TRAIN.STACK):
                    offsets[i].append(o[i])
            else:
                h, s = kpn(z_f, x_f)
            for i in range(cfg.TRAIN.STACK):
                heatmap[i].append(h[i])
                objsize[i].append(s[i])

        if self.weighted:
            heatmap_weight = F.softmax(self.heatmap_weight, 0)
            if cfg.TRAIN.OFFSETS:
                offsets_weight = F.softmax(self.offsets_weight, 0)
            objsize_weight = F.softmax(self.objsize_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            out_heatmap = [weighted_avg(hm, heatmap_weight) for hm in heatmap]
            out_objsize = [weighted_avg(ob, objsize_weight) for ob in objsize]
            if cfg.TRAIN.OFFSETS:
                out_offsets = [weighted_avg(of, offsets_weight) for of in offsets]
                return out_heatmap, out_objsize, out_offsets
            else:
                return out_heatmap, out_objsize
        else:
            out_heatmap = [avg(hm) for hm in heatmap]
            out_objsize = [avg(ob) for ob in objsize]
            if cfg.TRAIN.OFFSETS:
                out_offsets = [avg(of) for of in offsets]
                return out_heatmap, out_objsize, out_offsets
            else:
                return out_heatmap, out_objsize
