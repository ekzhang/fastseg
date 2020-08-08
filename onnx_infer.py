"""Inference test with ONNX."""

import onnxruntime

import numpy as np
import torch
from torchvision import transforms

from PIL import Image

from fastseg.image import colorize, blend

im_path = 'c:/Users/ericzhang/Downloads/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_030669_leftImg8bit.png'
img = Image.open(im_path)

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ipt = torch.stack([tfms(img)]).numpy()

sess = onnxruntime.InferenceSession('mobilenetv3.onnx')

print(sess.get_inputs()[0].name)
out_ort = sess.run(None, {
    'input0': ipt,
})

labels = np.argmax(out_ort[0], axis=1)[0]
print(labels)

colorized = colorize(labels)
colorized.show()

composited = blend(img, colorized)
composited.show()
