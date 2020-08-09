"""Script to test inference of an exported ONNX model."""

import argparse

import numpy as np
import onnxruntime
import torch

from torchvision import transforms
from PIL import Image

from fastseg.image import colorize, blend

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('model', metavar='MODEL',
    help='filename of onnx model (e.g., mobilenetv3_large.onnx)')
parser.add_argument('image', metavar='IMAGE',
    help='filename of image to run inference on')

args = parser.parse_args()

im_path = args.image
img = Image.open(im_path)

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ipt = torch.stack([tfms(img)]).numpy()

sess = onnxruntime.InferenceSession(args.model)

out_ort = sess.run(None, {
    'input0': ipt,
})

labels = np.argmax(out_ort[0], axis=1)[0]
print(labels)

colorized = colorize(labels)
colorized.show()

composited = blend(img, colorized)
composited.show()
