"""Inference test code"""

import torch
import torch.nn as nn
from PIL import Image

from .model.lraspp import MobileV3Large
from .image.colorize import colorize_mask, blend

ckpt_path = 'c:/Users/ericzhang/Downloads/best_checkpoint_ep161.pth'
net = MobileV3Large.from_pretrained(ckpt_path).cuda().eval()

im_path = 'c:/Users/ericzhang/Downloads/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_030669_leftImg8bit.png'
img = Image.open(im_path)

seg = net.predict_one(img)

colorized = colorize_mask(seg)
colorized.show()

composited = blend(img, colorized)
composited.show()
