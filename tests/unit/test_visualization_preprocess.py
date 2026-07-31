"""Tests for OpenCV preview preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from hypersat.exceptions import PreviewError
from hypersat.visualization.preprocess import (
    gaussian_blur,
    resize_to_max_dimension,
    target_preview_shape,
)


def test_target_preview_shape_leaves_small_images_alone() -> None:
    assert target_preview_shape(800, 600, 2048) == (600, 800)


def test_target_preview_shape_shrinks_the_long_side() -> None:
    height, width = target_preview_shape(4000, 2000, 1000)

    assert max(height, width) == 1000
    assert width == 1000
    assert height == 500


def test_resize_downsamples_with_area_interpolation() -> None:
    image = np.arange(100 * 80, dtype=np.uint8).reshape(100, 80)

    resized = resize_to_max_dimension(image, 50)

    assert resized.shape == (50, 40)


def test_gaussian_blur_requires_an_odd_kernel() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(PreviewError, match="odd kernel"):
        gaussian_blur(image, 4)


def test_gaussian_blur_preserves_shape() -> None:
    image = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)

    blurred = gaussian_blur(image, 5)

    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
