"""Modified EfficientNets for use as semantic segmentation feature extractors."""

import torch
import torch.nn as nn

from geffnet import tf_efficientnet_b4, tf_efficientnet_b0


class EfficientNet_B4(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=nn.BatchNorm2d,
                 pretrained=False):
        super(EfficientNet_B4, self).__init__()
        net = tf_efficientnet_b4(pretrained=pretrained,
                                 drop_rate=0.25,
                                 drop_connect_rate=0.2,
                                 norm_layer=BatchNorm)

        self.output_stride = output_stride
        self.early = nn.Sequential(net.conv_stem,
                                   net.bn1,
                                   net.act1)
        if self.output_stride == 8:
            block3_stride = 1
            block5_stride = 1
            dilation_blocks34 = 2
            dilation_blocks56 = 4
        elif self.output_stride == 16:
            block3_stride = 1
            block5_stride = 2
            dilation_blocks34 = 1
            dilation_blocks56 = 1
        else:
            raise

        net.blocks[3][0].conv_dw.stride = (block3_stride, block3_stride)
        net.blocks[5][0].conv_dw.stride = (block5_stride, block5_stride)

        for block_num in (3, 4, 5, 6):
            for sub_block in range(len(net.blocks[block_num])):
                m = net.blocks[block_num][sub_block].conv_dw
                if block_num < 5:
                    m.dilation = (dilation_blocks34, dilation_blocks34)
                    pad = dilation_blocks34
                else:
                    m.dilation = (dilation_blocks56, dilation_blocks56)
                    pad = dilation_blocks56
                if m.kernel_size[0] == 3:
                    pad *= 1
                elif m.kernel_size[0] == 5:
                    pad *= 2
                else:
                    raise
                m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.late = nn.Sequential(net.conv_head,
                                  net.bn2,
                                  net.act2)
        del net

    def forward(self, x):
        x = self.early(x)
        x = self.block0(x)
        s2 = x
        x = self.block1(x)
        s4 = x
        x = self.block2(x)
        s8 = x
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.late(x)
        if self.output_stride == 8:
            return s2, s4, x
        else:
            return s4, s8, x


class EfficientNet_B0(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=nn.BatchNorm2d,
                 pretrained=False):
        super(EfficientNet_B0, self).__init__()
        net = tf_efficientnet_b0(pretrained=pretrained,
                                 drop_rate=0.25,
                                 drop_connect_rate=0.2,
                                 norm_layer=BatchNorm)

        self.output_stride = output_stride
        self.early = nn.Sequential(net.conv_stem,
                                   net.bn1,
                                   net.act1)
        if self.output_stride == 8:
            block3_stride = 1
            block5_stride = 1
            dilation_blocks34 = 2
            dilation_blocks56 = 4
        elif self.output_stride == 16:
            block3_stride = 1
            block5_stride = 2
            dilation_blocks34 = 1
            dilation_blocks56 = 1
        else:
            raise

        net.blocks[3][0].conv_dw.stride = (block3_stride, block3_stride)
        net.blocks[5][0].conv_dw.stride = (block5_stride, block5_stride)

        for block_num in (3, 4, 5, 6):
            for sub_block in range(len(net.blocks[block_num])):
                m = net.blocks[block_num][sub_block].conv_dw
                if block_num < 5:
                    m.dilation = (dilation_blocks34, dilation_blocks34)
                    pad = dilation_blocks34
                else:
                    m.dilation = (dilation_blocks56, dilation_blocks56)
                    pad = dilation_blocks56
                if m.kernel_size[0] == 3:
                    pad *= 1
                elif m.kernel_size[0] == 5:
                    pad *= 2
                else:
                    raise
                m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.late = nn.Sequential(net.conv_head,
                                  net.bn2,
                                  net.act2)
        del net

    def forward(self, x):
        x = self.early(x)
        x = self.block0(x)
        s2 = x
        x = self.block1(x)
        s4 = x
        x = self.block2(x)
        s8 = x
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.late(x)
        if self.output_stride == 8:
            return s2, s4, x
        else:
            return s4, s8, x


if __name__ == "__main__":
    model = EfficientNet_B0(BatchNorm=nn.BatchNorm2d, pretrained=True,
                            output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    low, mid, x = model(input)
    print(model)
    print(sum(p.numel() for p in model.parameters()), ' parameters')
    print(x.size())
    print(low.size())
    print(mid.size())
