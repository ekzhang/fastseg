# Fast Semantic Segmentation

This respository contains real-time semantic segmentation code, trained on Cityscapes. It can be used for efficient inference on street images from similar datasets, such as KITTI, Mapillary Vistas, or the like.

![Example image segmentation](https://i.imgur.com/WspmlwN.jpg)

The architecture used is an implementation of MobileNetV3 with a modified segmentation head based on LR-ASPP. We are able to achieve 71.4% mIOU on Cityscapes _val_, while running at 60 FPS on a GPU. Please see below for detailed benchmarks.

## Requirements

This code requires Python 3.7 or later. It is tested with PyTorch version 1.5 and above. To install the package, simply run `pip install fastseg`. You will then be able to import and use the package like so:

```python
from fastseg import MobileV3Large
model = MobileV3Large.from_pretrained('path_to_checkpoint.pth')
model.eval()
```

More detailed examples are given below. Alternatively, to use the code from source, clone this repository and install the `geffnet` package (and additional dependencies) by running `pip install -r requirements.txt` in your Python environment.

## Example: Running Inference

Currently you can test inference of a dummy model by running `python -m fastseg.infer` in the project root.

## Example: Exporting to ONNX

TODO

## Training from Scratch

TODO

## Acknowledgements

Special thanks to Andrew Tao and Karan Sapra from [NVIDIA ADLR](https://nv-adlr.github.io/) for many helpful discussions, as well as for sharing their training code with me (you can access this at [NVIDIA/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)). Also, thanks to Branislav Kisacanin who introduced me to Nvidia and put semantic segmentation in the big picture.
