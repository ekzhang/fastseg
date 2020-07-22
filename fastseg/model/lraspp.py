"""Lite Reduced Atrous Spatial Pyramid Pooling

Architecture introduced in the MobileNetV3 (2019) paper, as an
efficient semantic segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_trunk, ConvBnRelu

class LRASPP(nn.Module):
    """Lite R-ASPP Segmentation Network"""
    def __init__(self,
                 num_classes,
                 trunk='mobilenetv3_large',
                 criterion=None,
                 hidden_ch=128,
                 bottleneck_ch=256):
        super(LRASPP, self).__init__()

        self.criterion = criterion
        self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk)
        
        # Reduced atrous spatial pyramid pooling
        self.aspp_conv1 = nn.Sequential(
            nn.Conv2d(high_level_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.aspp_conv2 = nn.Sequential(
            nn.Conv2d(high_level_ch, bottleneck_ch, 1, bias=False),
            nn.Conv2d(bottleneck_ch, hidden_ch, 3, dilation=12, padding=12),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.aspp_conv3 = nn.Sequential(
            nn.Conv2d(high_level_ch, bottleneck_ch, 1, bias=False),
            nn.Conv2d(bottleneck_ch, hidden_ch, 3, dilation=36, padding=36),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_level_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        aspp_out_ch = hidden_ch * 4

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, hidden_ch, kernel_size=1, bias=False)
        self.conv_up2 = ConvBnRelu(hidden_ch + 64, hidden_ch, kernel_size=5, padding=2)
        self.conv_up3 = ConvBnRelu(hidden_ch + 32, hidden_ch, kernel_size=5, padding=2)
        self.last = nn.Conv2d(hidden_ch, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        s2, s4, final = self.trunk(x)
        aspp = torch.cat([
            self.aspp_conv1(final),
            self.aspp_conv2(final),
            self.aspp_conv3(final),
            F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
        ], 1)
        y = self.conv_up1(aspp)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        return y


def MobileV3Large(num_classes, criterion):
    return LRASPP(num_classes, criterion=criterion, trunk='mobilenetv3_large')


def MobileV3Small(num_classes, criterion):
    return LRASPP(num_classes, criterion=criterion, trunk='mobilenetv3_small', bottleneck_ch=128)
