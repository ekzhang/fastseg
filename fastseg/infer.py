"""Inference test code"""

import torch
import torch.nn as nn
from PIL import Image

from .model.lraspp import MobileV3Large
from .image.colorize import colorize, blend

ckpt_path = 'c:/Users/ericzhang/Downloads/best_checkpoint_ep161.pth'
net = MobileV3Large.from_pretrained(ckpt_path).cuda().eval()

tot_bn = 0
li = []
for name, param in net.state_dict().items():
    li.append((param.numel(), name))
    if 'bn' in name or 'running_' in name:
        tot_bn += param.numel()
print(tot_bn)
li.sort(reverse=True)
for p, n in li:
    print(p, n)

import sys
sys.exit(0)

im_path = 'c:/Users/ericzhang/Downloads/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_030669_leftImg8bit.png'
img = Image.open(im_path)

seg = net.predict_one(img)

colorized = colorize(seg)
colorized.show()

composited = blend(img, colorized)
composited.show()
