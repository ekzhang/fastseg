# Fast Semantic Segmentation

This respository contains real-time semantic segmentation code, trained on Cityscapes. It can be used for efficient inference on street images from similar datasets, such as KITTI, Mapillary Vistas, or the like.

![Example image segmentation](https://i.imgur.com/WspmlwN.jpg)

The architecture used is an implementation of MobileNetV3 with a modified segmentation head based on LR-ASPP. We are able to achieve 71.4% mIOU on Cityscapes _val_, while running at 60 FPS on a GPU. Please see below for detailed benchmarks.

## Requirements

This code is tested with PyTorch version 1.5 and requires the `geffnet` package, as described in `requirements.txt`. To install, run `pip install -r requirements.txt` in your Python environment. 

## Running Inference

TODO

## Training from Scratch

TODO
