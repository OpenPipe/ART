"""
Utilities for working with image processors. Create new types to avoid needing to import AutoImageProcessor and BaseImageProcessor.


Avoid importing AutoImageProcessor and BaseImageProcessor until runtime, because they're slow imports.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

from PIL import Image

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.image_processing_utils import BaseImageProcessor

    ImageProcessor: TypeAlias = BaseImageProcessor
else:
    # make it importable from other files as a type in runtime
    ImageProcessor: TypeAlias = Any


@cache
def get_image_processor(model_name: str) -> ImageProcessor:
    model_name = model_name.split(":")[0]

    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    assert processor.is_fast, f"Could not load fast image processor for {model_name}"
    return processor


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resize an image so that its longest side is at most max_size pixels.

    Preserves aspect ratio and uses LANCZOS resampling for quality.
    Returns the original image if it's already smaller than max_size.
    """

    width, height = image.size
    if max(width, height) <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
