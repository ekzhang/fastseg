# Fast Semantic Segmentation

This respository aims to provide accurate _real-time semantic segmentation_ code for mobile devices, with pretrained weights on Cityscapes. This can be used for efficient segmentation on a variety of real-world street images, as well as images from other datasets like Mapillary Vistas, KITTI, and CamVid.

![Example image segmentation](https://i.imgur.com/WspmlwN.jpg)



The models are implementations of **MobileNetV3** (both large and small variants) with a modified segmentation head based on **LR-ASPP**. The top model was able to achieve **71.4%** mIOU on Cityscapes _val_, while running at up to **60 FPS** on a GPU. Please see below for detailed benchmarks.

Currently, you can do the following:

- Load pretrained MobileNetV3 semantic segmentation models.
- Easily generate hard segmentation labels or soft probabilities for street image scenes.
- Evaluate MobileNetV3 models on Cityscapes or your own dataset.
- Export models for production with ONNX.

If you have any feature requests or questions, feel free to leave them as GitHub issues!

## What's New?

### August X, 2020

- Initial release

## Requirements

This code requires Python 3.7 or later. It is tested with PyTorch version 1.5 or above. To install the package, simply run `pip install fastseg`. You can then load the pretrained model:

```python
# Load a pretrained MobileNetV3 segmentation model in inference mode
from fastseg import MobileV3Large
model = MobileV3Large.from_pretrained().cuda()
model.eval()

# Open a local image as input
from PIL import Image
image = Image.open('street_image.png')

# Preprocess the image and predict numeric labels [0-18] for each pixel
labels = model.predict_one(image)

# Display a color-coded output
from fastseg.image import colorize, blend
colorized = colorize(labels)
colorized.show()
composited = blend(image, colorized)
composited.show()
```

More detailed examples are given below. Alternatively, to use the code from source, clone this repository and install the `geffnet` package (along with additional dependencies) by running `pip install -r requirements.txt` in the project root.

## Pretrained Models and Metrics

I was able to train a few models close to or exceeding the accuracy described in the original [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper. Each was trained only on the `gtFine` labels from Cityscapes for around 12 hours on an Nvidia DGX-1 node, with 8 V100 GPUs.

| Model           | Segmentation Head           | Parameters | mIOU  | Inference | Pretrained? |
| --------------- | --------------------------- | ---------- | ----- | --------- | :---------: |
| `MobileV3Large` | LR-ASPP, 1x1 Up-Conv, F=256 | 3.6M       | 71.4% | 21.1 FPS  |      ✔      |
| `MobileV3Large` | LR-ASPP, 1x1 Up-Conv, F=128 | 3.2M       | 68.1% | 25.7 FPS  |             |
| `MobileV3Small` | LR-ASPP, 1x1 Up-Conv, F=256 | 1.4M       | 63.4% | 30.3 FPS  |      ✔      |

For comparison, the original paper reports 72.6% mIOU and 3.6M parameters on the Cityscapes _val_ set. Inference was done on a single Nvidia V100 GPU with 16-bit floating point precision, tested on full-resolution 2MP images (1024 x 2048) from Cityscapes as input. It is much faster for half-resolution (512 x 1024) images.

TODO: Get inference times with TensorRT/ONNX. I expect these to be significantly faster.

## Example: Running Inference

Currently you can test inference of a dummy model by running `python infer.py` in the project root.

TODO actual code and "Open in Colab" button, after this gets added to PyPI.

## Example: Exporting to ONNX

TODO

## Training from Scratch

TODO

## Contributions

Pull requests are always welcome! Thanks to Andrew Tao and Karan Sapra from [NVIDIA ADLR](https://nv-adlr.github.io/) for many helpful discussions, as well as Branislav Kisacanin, without whom this wouldn't be possible.

Licensed under the MIT License.
