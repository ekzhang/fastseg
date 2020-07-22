"""Inference test code"""

import torch
import torch.nn as nn
from geffnet.mobilenetv3 import tf_mobilenetv3_large_100

from .model.lraspp import MobileV3Large
from .model.mobilenetv3 import MobileNetV3_Large

net = MobileV3Large(19, None).cuda().eval()
# net = MobileNetV3_Large().cuda().eval()
# net = tf_mobilenetv3_large_100(drop_rate=0.2, norm_layer=nn.BatchNorm2d).cuda().eval()
# net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).cuda().eval()

data = torch.rand((20, 3, 1024, 2048)).cuda()

with torch.no_grad():
    for x in data:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = net(x.unsqueeze(0))
        end.record()
        torch.cuda.synchronize()
        print(f'{start.elapsed_time(end) / 1000.0:.3f}')
        print(out.shape)
