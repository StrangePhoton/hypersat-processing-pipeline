"""OpenCV helpers used by cosmetic previews.

Morphology for the quality mask lives in Milestone 6; this module is deliberately small:
downsampling a preview to a maximum display size, and an optional Gaussian blur whose
kernel size is always explicit. Nothing here opens a raster or writes a GeoTIFF.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy.typing as npt

from hypersat.exceptions import PreviewError

__all__ = [
    "gaussian_blur",
    "resize_to_max_dimension",
    "target_preview_shape",
]

_IMAGE_NDIM_GREY = 2
_IMAGE_NDIM_COLOR = 3


def target_preview_shape(
    width: int,
    height: int,
    max_dimension: int,
) -> tuple[int, int]:
    """Return ``(height, width)`` that fits inside ``max_dimension`` on the long side.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        max_dimension: Longest allowed side of the preview; must be positive.

    Returns:
        The (possibly unchanged) preview shape as ``(rows, columns)``.

    Raises:
        ValueError: If any argument is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width} x {height}")
    if max_dimension <= 0:
        raise ValueError(f"max_dimension must be positive, got {max_dimension}")
    longest = max(width, height)
    if longest <= max_dimension:
        return height, width
    scale = max_dimension / longest
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_to_max_dimension(
    image: npt.NDArray[Any],
    max_dimension: int,
) -> npt.NDArray[Any]:
    """Downsample an image so its longest side is at most ``max_dimension``.

    Uses area interpolation, which is the right choice for shrinking continuous imagery.
    Upsampling is never performed: a small source is returned unchanged.

    Args:
        image: ``(rows, columns)`` or ``(rows, columns, channels)`` array.
        max_dimension: Longest allowed side.

    Returns:
        The resized image, or ``image`` itself when already small enough.

    Raises:
        PreviewError: If the array rank is unsupported.
        ValueError: If ``max_dimension`` is not positive.
    """
    if image.ndim not in (_IMAGE_NDIM_GREY, _IMAGE_NDIM_COLOR):
        raise PreviewError(
            "Preview images must be 2- or 3-dimensional.",
            hint="Pass a single-band (H, W) array or an (H, W, C) composite.",
            context={"ndim": int(image.ndim), "shape": [int(size) for size in image.shape]},
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    target_height, target_width = target_preview_shape(width, height, max_dimension)
    if (target_height, target_width) == (height, width):
        return image
    resized: npt.NDArray[Any] = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def gaussian_blur(
    image: npt.NDArray[Any],
    kernel_size: int,
) -> npt.NDArray[Any]:
    """Apply a Gaussian blur with an explicit odd kernel size.

    Args:
        image: ``(rows, columns)`` or ``(rows, columns, channels)`` array.
        kernel_size: Odd positive kernel edge length in pixels. There is no default: blur
            is never applied by accident.

    Returns:
        The blurred image.

    Raises:
        PreviewError: If ``kernel_size`` is not a positive odd integer, or the array rank
            is unsupported.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise PreviewError(
            "Gaussian blur requires a positive odd kernel size.",
            hint="Pass an odd value such as 3, 5 or 7; omit blur entirely when none is wanted.",
            context={"kernel_size": kernel_size},
        )
    if image.ndim not in (_IMAGE_NDIM_GREY, _IMAGE_NDIM_COLOR):
        raise PreviewError(
            "Preview images must be 2- or 3-dimensional.",
            hint="Pass a single-band (H, W) array or an (H, W, C) composite.",
            context={"ndim": int(image.ndim), "shape": [int(size) for size in image.shape]},
        )
    blurred: npt.NDArray[Any] = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=0.0, sigmaY=0.0
    )
    return blurred
