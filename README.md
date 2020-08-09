# Fast Semantic Segmentation

This respository aims to provide accurate _real-time semantic segmentation_ code for mobile devices in PyTorch, with pretrained weights on Cityscapes. This can be used for efficient segmentation on a variety of real-world street images, including datasets like Mapillary Vistas, KITTI, and CamVid.

```python
from fastseg import MobileV3Large
model = MobileV3Large.from_pretrained().cuda().eval()

model.predict(images)
```

![Example image segmentation video](https://i.imgur.com/vOApT8N.gif)

The models are implementations of **MobileNetV3** (both large and small variants) with a modified segmentation head based on **LR-ASPP**. The top model was able to achieve **72.3%** mIOU on Cityscapes _val_, while running at up to **30.7 FPS** on a GPU. Please see below for detailed benchmarks.

Currently, you can do the following:

- Load pretrained MobileNetV3 semantic segmentation models.
- Easily generate hard segmentation labels or soft probabilities for street image scenes.
- Evaluate MobileNetV3 models on Cityscapes, or your own dataset.
- Export models for production with ONNX.

If you have any feature requests or questions, feel free to leave them as GitHub issues!

## Table of Contents

  * [What's New?](#whats-new)
    + [August X, 2020](#august-x-2020)
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Pretrained Models and Metrics](#pretrained-models-and-metrics)
  * [Usage](#usage)
    + [Example: Running Inference](#example-running-inference)
    + [Example: Exporting to ONNX](#example-exporting-to-onnx)
  * [Training from Scratch](#training-from-scratch)
  * [Contributions](#contributions)

## What's New?

### August X, 2020

- Initial release

## Overview

Here's an excerpt from the [original paper](https://arxiv.org/abs/1905.02244) introducing MobileNetV3:

> This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small, which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation.
>
> For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). **We achieve new state of the art results for mobile classification, detection and segmentation.**
>
> **MobileNetV3-Large LRASPP is 34% faster than MobileNetV2 R-ASPP at similar
accuracy for Cityscapes segmentation.**
>
> ![MobileNetV3 Comparison](https://i.imgur.com/E9IYp0c.png?1)

This project tries to faithfully implement MobileNetV3 for real-time semantic segmentation, with the aims of being efficient, easy to use, and extendable.

## Requirements

This code requires Python 3.7 or later. It is tested with PyTorch version 1.5 and above. To install the package, simply run `pip install fastseg`. Then you can get started with a pretrained model:

```python
# Load a pretrained MobileNetV3 segmentation model in inference mode
from fastseg import MobileV3Large
model = MobileV3Large.from_pretrained().cuda()
model.eval()

# Open a local image as input
from PIL import Image
image = Image.open('street_image.png')

# Predict numeric labels [0-18] for each pixel of the image
labels = model.predict_one(image)
```

![Example image segmentation](https://i.imgur.com/WspmlwN.jpg)

More detailed examples are given below. As an alternative, instead of installing `fastseg` from pip, you can clone this repository and install the `geffnet` package (along with other dependencies) by running `pip install -r requirements.txt` in the project root.

## Pretrained Models and Metrics

I was able to train a few models close to or exceeding the accuracy described in the original [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper. Each was trained only on the `gtFine` labels from Cityscapes for around 12 hours on an Nvidia DGX-1 node, with 8 V100 GPUs.

| Model           | Segmentation Head | Parameters | mIOU  | Inference | TensorRT | Weights? |
| --------------- | ----------------- | ---------- | ----- | --------- | -------- | :------: |
| `MobileV3Large` | LR-ASPP, F=256    | 3.6M       | 72.3% | 21.1 FPS  | 30.7 FPS |    ✔     |
| `MobileV3Large` | LR-ASPP, F=128    | 3.2M       | 68.1% | 25.7 FPS  | --       |          |
| `MobileV3Small` | LR-ASPP, F=256    | 1.4M       | 66.5% | 30.3 FPS  | 39.4 FPS |    ✔     |

For comparison, this is within 0.3% of the original paper, which reports 72.6% mIOU and 3.6M parameters on the Cityscapes _val_ set. Inference was done on an Nvidia V100 GPU, tested on full-resolution 2MP images (1024 x 2048) from Cityscapes as input. It runs much faster on half-resolution (512 x 1024) images.

The "TensorRT" column shows some benchmarks I ran while experimenting with exporting optimized ONNX models to [Nvidia TensorRT](https://developer.nvidia.com/tensorrt). You might be able to get additional speedups if you're knowledgeable about this.

## Usage

### Example: Running Inference

The easiest way to get started with inference is to clone this repository and use the `infer.py` script. For example, if you have street images named `city_1.png` and `city_2.png`, then you can generate segmentation labels for them with the following command.

```shell
$ python infer.py city_1.png city_2.png
```

Output:
```
==> Creating PyTorch MobileV3Large model
==> Loading images and running inference
Loading city_1.png
Generated colorized_city_1.png
Generated composited_city_1.png
Loading city_2.png
Generated colorized_city_2.png
Generated composited_city_2.png
```

|               Original               |              Colorized               |              Composited              |
| :----------------------------------: | :----------------------------------: | :----------------------------------: |
| ![](https://i.imgur.com/74vqz0q.png) | ![](https://i.imgur.com/HRr16YC.png) | ![](https://i.imgur.com/WVd5a6Z.png) |
| ![](https://i.imgur.com/MJA7VMN.png) | ![](https://i.imgur.com/FqoxHzR.png) | ![](https://i.imgur.com/fVMvbRv.png) |

To interact with the models programmatically, first install the `fastseg` package with pip, as described above. Then, you can import and construct the models in your own Python code, which are normal instances of PyTorch `nn.Module`.

```python
from fastseg import MobileV3Large, MobileV3Small

# Construct a new model with pretrained weights
model = MobileV3Large.from_pretrained()

# Construct a new model from a local .pth checkpoint
model = MobileV3Small.from_pretrained('path_to_weights.pth')

# Construct a custom model with random initialization
model = MobileV3Large(
    num_classes=19,
    trunk='mobilenetv3_large',
    use_aspp=False,
    hidden_ch=256,
)
```

To run inference on an image or batch of images, you can use the methods `model.predict_one()` and `model.predict()`, respectively. These methods take care of the preprocessing and output interpretation for you; they take PIL Images or NumPy arrays as input and return a NumPy array.

(You can also run inference directly with `model.forward()`, which will return a tensor containing logits, but be sure to normalize the inputs to have mean 0 and variance 1.)

```python
import torch
from PIL import Image
from fastseg import MobileV3Large, MobileV3Small

# Construct a new model with pretrained weights, in evaluation mode
model = MobileV3Large.from_pretrained().cuda()
model.eval()

# Run inference on an image
img = Image.open('city_1.png')
labels = model.predict_one(img) # returns a NumPy array containing integer labels
assert labels.shape == (224, 224)

# Run inference on a batch of images
img2 = Image.open('city_2.png')
batch_labels = model.predict([img, img2]) # returns a NumPy array containing integer labels
assert batch_labels.shape == (2, 224, 224)

# Run inference directly
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
with torch.no_grad():
    dummy_output = model(dummy_input)
assert dummy_output.shape == (1, 19, 224, 224)
```

In addition, you can generate colorized and composited versions of the label masks as human-interpretable images.

```python
from fastseg.image import colorize, blend

colorized = colorize(labels) # returns a PIL Image
colorized.show()

composited = blend(img, colorized) # returns a PIL Image
composited.show()
```

You can see an example of this in action by pressing the "Open in Colab" button below.

TODO "Open in Colab" button & notebook.

### Example: Exporting to ONNX

The script `onnx_export.py` can be used to convert a pretrained segmentation model to ONNX. You should specify the image input dimensions when exporting. See the usage instructions below:

```
$ python onnx_export.py --help
usage: onnx_export.py [-h] [--model MODEL] [--size SIZE]
                      [--checkpoint CHECKPOINT]
                      OUTPUT_FILENAME

Command line script to export a pretrained segmentation model to ONNX.

positional arguments:
  OUTPUT_FILENAME       filename of output model (e.g.,
                        mobilenetv3_large.onnx)

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        the model to export (default MobileV3Large)
  --size SIZE, -s SIZE  the image dimensions to set as input (default
                        1024,2048)
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        filename of the weights checkpoint .pth file (uses
                        pretrained by default)
```

We also provide an `onnx_optimize.py` script for optimizing exported models.

## Training from Scratch

Coming soon!

## Contributions

Pull requests are always welcome! A big thanks to Andrew Tao and Karan Sapra from [NVIDIA ADLR](https://nv-adlr.github.io/) for helpful discussions and for lending me their training code, as well as Branislav Kisacanin, without whom this wouldn't be possible.

Licensed under the MIT License.
