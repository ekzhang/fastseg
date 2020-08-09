# Fast Semantic Segmentation

This respository aims to provide accurate _real-time semantic segmentation_ code for mobile devices, with pretrained weights on Cityscapes. This can be used for efficient segmentation on a variety of real-world street images, as well as images from other datasets like Mapillary Vistas, KITTI, and CamVid.

![Example image segmentation](https://i.imgur.com/WspmlwN.jpg)

The models are implementations of **MobileNetV3** (both large and small variants) with a modified segmentation head based on **LR-ASPP**. The top model was able to achieve **71.4%** mIOU on Cityscapes _val_, while running at up to **60 FPS** on a GPU. Please see below for detailed benchmarks.

Currently, you can do the following:

- Load pretrained MobileNetV3 semantic segmentation models.
- Easily generate hard segmentation labels or soft probabilities for street image scenes.
- Evaluate MobileNetV3 models on Cityscapes, or your own dataset.
- Export models for production with ONNX.

If you have any feature requests or questions, feel free to leave them as GitHub issues!

## Table of Contents

  * [What's New?](#whats-new)
    + [August X, 2020](#august-x-2020)
  * [About MobileNetV3](#about-mobilenetv3)
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

## About MobileNetV3

Here's an excerpt from the abstract of the [original paper](https://arxiv.org/abs/1905.02244):

> This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small, which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation.
>
> For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP).
>
> **MobileNetV3-Large LRASPP is 34% faster than MobileNetV2 R-ASPP at similar
accuracy for Cityscapes segmentation.**
>
> ![MobileNetV3 Comparison](https://i.imgur.com/E9IYp0c.png?1)

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

| Model           | Segmentation Head | Parameters | mIOU  | Inference | Weights? |
| --------------- | ----------------- | ---------- | ----- | --------- | :------: |
| `MobileV3Large` | LR-ASPP, F=256    | 3.6M       | 71.4% | 21.1 FPS  |    ✔     |
| `MobileV3Large` | LR-ASPP, F=128    | 3.2M       | 68.1% | 25.7 FPS  |          |
| `MobileV3Small` | LR-ASPP, F=256    | 1.4M       | 63.4% | 30.3 FPS  |    ✔     |

For comparison, the original paper reports 72.6% mIOU and 3.6M parameters on the Cityscapes _val_ set. Inference was done on a single Nvidia V100 GPU with 16-bit floating point precision, tested on full-resolution 2MP images (1024 x 2048) from Cityscapes as input. It is much faster for half-resolution (512 x 1024) images.

TODO: Get inference times with TensorRT/ONNX. I expect these to be significantly faster.

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

You can run raw inference in your own pipeline with `model.forward()`, like any other PyTorch model. However, we also provide convenience functions `model.predict_one()` and `model.predict()`, which run preprocessing and normalization on PIL images directly and return labels.

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

## Training from Scratch

Coming soon!

## Contributions

Pull requests are always welcome! A big thanks to Andrew Tao and Karan Sapra from [NVIDIA ADLR](https://nv-adlr.github.io/) for helpful discussions and for lending me their training code, as well as Branislav Kisacanin, without whom this wouldn't be possible.

Licensed under the MIT License.
