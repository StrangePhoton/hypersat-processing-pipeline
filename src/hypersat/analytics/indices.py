"""Normalised-difference spectral indices as pure array functions.

NDVI and NDWI are conventional broadband indices, not hyperspectral algorithms. On a
hyperspectral product the two input bands are chosen by wavelength elsewhere; this module
only does the arithmetic:

* NDVI = (NIR - Red) / (NIR + Red)
* NDWI = (Green - NIR) / (Green + NIR)   (McFeeters)

Division by zero, non-finite samples and masked NoData never become silent "valid" index
values: they are written as the configured NoData.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_INDEX_NODATA",
    "ndvi",
    "ndwi",
    "normalised_difference",
]

DEFAULT_INDEX_NODATA = -9999.0
"""NoData written into float32 index GeoTIFFs; matches ``configs/pipeline.example.yaml``."""


def normalised_difference(
    minuend: npt.NDArray[Any],
    subtrahend: npt.NDArray[Any],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    nodata: float = DEFAULT_INDEX_NODATA,
) -> npt.NDArray[np.float32]:
    """Compute ``(minuend - subtrahend) / (minuend + subtrahend)`` safely.

    Args:
        minuend: Numerator's positive term (NIR for NDVI, green for NDWI).
        subtrahend: Numerator's subtracted term (red for NDVI, NIR for NDWI).
        mask: Optional boolean mask with the same shape; ``True`` means NoData.
        nodata: Value written where the index is undefined.

    Returns:
        A ``float32`` array of the same shape. Valid results lie in approximately
        ``[-1, 1]``; undefined locations hold ``nodata``.

    Raises:
        ValueError: If the arrays (and mask) do not share one shape.
    """
    left = np.asarray(minuend, dtype=np.float64)
    right = np.asarray(subtrahend, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"band arrays must share a shape; got {left.shape} and {right.shape}")
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != left.shape:
            raise ValueError(
                f"mask shape {mask_array.shape} does not match band shape {left.shape}"
            )
    else:
        mask_array = None

    numerator = left - right
    denominator = left + right
    valid = np.isfinite(left) & np.isfinite(right) & (denominator != 0.0)
    if mask_array is not None:
        valid &= ~mask_array

    result = np.full(left.shape, nodata, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def ndvi(
    nir: npt.NDArray[Any],
    red: npt.NDArray[Any],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    nodata: float = DEFAULT_INDEX_NODATA,
) -> npt.NDArray[np.float32]:
    """Compute the Normalised Difference Vegetation Index.

    Args:
        nir: Near-infrared band.
        red: Red band.
        mask: Optional NoData mask shared by both bands.
        nodata: Value written where the index is undefined.

    Returns:
        Float32 NDVI array.
    """
    return normalised_difference(nir, red, mask=mask, nodata=nodata)


def ndwi(
    green: npt.NDArray[Any],
    nir: npt.NDArray[Any],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    nodata: float = DEFAULT_INDEX_NODATA,
) -> npt.NDArray[np.float32]:
    """Compute McFeeters' Normalised Difference Water Index.

    Args:
        green: Green band.
        nir: Near-infrared band.
        mask: Optional NoData mask shared by both bands.
        nodata: Value written where the index is undefined.

    Returns:
        Float32 NDWI array.
    """
    return normalised_difference(green, nir, mask=mask, nodata=nodata)
