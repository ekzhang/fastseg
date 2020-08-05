"""Utilities for generating a colorized segmentation image.

Parts of this code were modified from https://github.com/NVIDIA/semantic-segmentation.
"""

import numpy as np
from PIL import Image

from .palette import all_palettes

def colorize_mask(mask_array, palette='cityscapes'):
    """Colorize a segmentation mask.

    Keyword arguments:
    mask_array -- the segmentation as a 2D numpy array of integers [0..classes - 1]
    palette -- the palette to use (default 'cityscapes')
    """
    mask_img = Image.fromarray(mask_array.astype(np.uint8)).convert('P')
    mask_img.putpalette(all_palettes[palette])
    return mask_img.convert('RGB')

def blend(input_img, seg_img):
    """Blend an input image with its colorized segmentation labels."""
    return Image.blend(input_img, seg_img, 0.4)
