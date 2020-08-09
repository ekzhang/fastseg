"""Command line script to test inference on one or more images."""

import argparse
import os.path as path
import sys

import torch
import torch.nn as nn
from PIL import Image

from fastseg import MobileV3Large, MobileV3Small
from fastseg.image import colorize, blend

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('images', nargs='*', metavar='IMAGES',
    help='one or more filenames of images to run inference on')
parser.add_argument('--model', '-m', default='MobileV3Large',
    help='the model to use for inference (default MobileV3Large)')
parser.add_argument('--checkpoint', '-c', default=None,
    help='filename of the weights checkpoint .pth file (uses pretrained by default)')
parser.add_argument('--show', '-s', action='store_true',
    help='display the output segmentation results in the default image viewer')

args = parser.parse_args()

if not args.images:
    print('Please supply at least one image to run inference on.', file=sys.stderr)
    sys.exit(1)

print(f'==> Creating PyTorch {args.model} model')
if args.model == 'MobileV3Large':
    model_cls = MobileV3Large
elif args.model == 'MobileV3Small':
    model_cls = MobileV3Small
else:
    print(f'Unknown model name: {args.model}', file=sys.stderr)
    sys.exit(1)

model = model_cls.from_pretrained(args.checkpoint).cuda().eval()

print('==> Loading images and running inference')

for im_path in args.images:
    print('Loading', im_path)
    img = Image.open(im_path)

    seg = model.predict_one(img)

    colorized = colorize(seg)
    composited = blend(img, colorized)

    basename, filename = path.split(im_path)
    colorized_filename = 'colorized_' + filename
    composited_filename = 'composited_' + filename
    colorized.save(colorized_filename)
    composited.save(composited_filename)
    print(f'Generated {colorized_filename}')
    print(f'Generated {composited_filename}')

    if args.show:
        colorized.show()
        composited.show()
