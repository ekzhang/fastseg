"""Modified MobileNetV3 for use as semantic segmentation feature extractors."""

import torch
import torch.nn as nn

from geffnet import tf_mobilenetv3_large_100, tf_mobilenetv3_small_100
from geffnet.efficientnet_builder import InvertedResidual, Conv2dSame, Conv2dSameExport

class MobileNetV3_Large(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_large_100, pretrained=False):
        super(MobileNetV3_Large, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[3][0].conv_dw.stride = (1, 1)
        net.blocks[5][0].conv_dw.stride = (1, 1)

        for block_num in (3, 4, 5, 6):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 5:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]

    def forward(self, x):
        x = self.early(x) # 2x
        x = self.block0(x)
        s2 = x
        x = self.block1(x) # 4x
        s4 = x
        x = self.block2(x) # 8x
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return s2, s4, x


class MobileNetV3_Small(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_small_100, pretrained=False):
        super(MobileNetV3_Small, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[2][0].conv_dw.stride = (1, 1)
        net.blocks[4][0].conv_dw.stride = (1, 1)

        for block_num in (2, 3, 4, 5):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 4:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                assert m.kernel_size[0] == m.kernel_size[1]
                pad *= (m.kernel_size[0] - 1) // 2
                m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]

    def forward(self, x):
        x = self.early(x) # 2x
        s2 = x
        x = self.block0(x) # 4x
        s4 = x
        x = self.block1(x) # 8x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return s2, s4, x


if __name__ == '__main__':
    model = MobileNetV3_Large(pretrained=True)
    input = torch.rand(1, 3, 512, 512)
    low, mid, x = model(input)
    print(model)
    print(sum(p.numel() for p in model.parameters()), ' parameters')
    print(x.size())
    print(low.size())
    print(mid.size())
