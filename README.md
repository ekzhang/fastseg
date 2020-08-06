# Fast Semantic Segmentation

This respository aims to provide accurate real-time semantic segmentation code for mobile devices, with pretrained weights on Cityscapes. This can be used for efficient segmentation on a variety of real-world street images, as well as images from other datasets like Mapillary Vistas, KITTI, and CamVid.

![Example image segmentation](https://i.imgur.com/WspmlwN.jpg)

The models are implementations of MobileNetV3 (both large and small variants) with a modified segmentation head based on LR-ASPP. The top model was able to achieve 71.4% mIOU on Cityscapes _val_, while running at 60 FPS on a GPU. Please see below for detailed benchmarks.

## Requirements

This code requires Python 3.7 or later. It is tested with PyTorch version 1.5 and above. To install the package, simply run `pip install fastseg`. You will then be able to import and load the pretrained model:

```python
from fastseg import MobileV3Large
model = MobileV3Large.from_pretrained('path_to_checkpoint.pth')
model.eval()
```

More detailed examples are given below. Alternatively, to use the code from source, clone this repository and install the `geffnet` package (along with additional dependencies) by running `pip install -r requirements.txt` in the project root.

## Pretrained Models and Metrics

I was able to train a few of the models close to or exceeding the accuracy described in the original [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper.

| Model | Num Parameters | mIOU on Cityscapes _val_ | Inference Time |
| ----- | -------------- | ------------------------ | -------------- |
| MobileV3Large | 4.2M | 69.0% | xxx FPS |
| MobileV3Small | xxx | xxx% | xxx FPS |

Inference was done on a single Nvidia V100 GPU with 16-bit floating point precision, testing on full-resolution 2MP images (1024 x 2048) from Cityscapes as input. The inference time is much lower for half-resolution images.

## Example: Running Inference

Currently you can test inference of a dummy model by running `python -m fastseg.infer` in the project root.

## Example: Exporting to ONNX

TODO

## Training from Scratch

TODO

## Acknowledgements

Thanks to Andrew Tao and Karan Sapra from [NVIDIA ADLR](https://nv-adlr.github.io/) for many helpful discussions, as well as Branislav Kisacanin, without whom this wouldn't be possible.

Licensed under the MIT License.
