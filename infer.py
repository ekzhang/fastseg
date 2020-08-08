"""Command line script to test inference on a single image."""

import torch
import torch.nn as nn
from PIL import Image

from fastseg import MobileV3Large, MobileV3Small
from fastseg.image import colorize, blend

torch.backends.cudnn.benchmark = True

ckpt_path = 'c:/Users/ericzhang/Downloads/last_checkpoint_ep104.pth'
net = MobileV3Large.from_pretrained(ckpt_path).cuda().eval()

im_path = 'c:/Users/ericzhang/Downloads/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_030669_leftImg8bit.png'
img = Image.open(im_path)

seg = net.predict_one(img)

colorized = colorize(seg)
colorized.show()

composited = blend(img, colorized)
composited.show()
