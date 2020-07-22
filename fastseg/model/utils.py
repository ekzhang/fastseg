import torch.nn as nn

from .efficientnet import EfficientNet_B4, EfficientNet_B0
from .mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small


def get_trunk(trunk_name):
    """Retrieve the pretrained network trunk and channel counts"""

    if trunk_name == 'efficientnet_b4':
        backbone = EfficientNet_B4(pretrained=True)
        s2_ch = 24
        s4_ch = 32
        high_level_ch = 1792
    elif trunk_name == 'efficientnet_b0':
        backbone = EfficientNet_B0(pretrained=True)
        s2_ch = 16
        s4_ch = 24
        high_level_ch = 1280
    elif trunk_name == 'mobilenetv3_large':
        backbone = MobileNetV3_Large(pretrained=True)
        s2_ch = 16
        s4_ch = 24
        high_level_ch = 1280
    elif trunk_name == 'mobilenetv3_small':
        backbone = MobileNetV3_Small(pretrained=True)
        s2_ch = 16
        s4_ch = 48
        high_level_ch = 1024
    else:
        raise ValueError('unknown backbone {}'.format(trunk_name))

    return backbone, s2_ch, s4_ch, high_level_ch


class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.

    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
