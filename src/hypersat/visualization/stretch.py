"""Percentile contrast stretching for cosmetic previews.

Stretching lives here, and only here, so it cannot leak into a GeoTIFF product. Scientific
stages work on the source values; previews map a chosen percentile range onto 8-bit display
levels. Min/max stretching is deliberately absent: hyperspectral radiance usually has a
bright tail (specular water, clouds) that would collapse the useful range to a handful of
grey levels.

NoData is metadata. Percentiles are computed on valid samples only, and masked positions
become black (0) in the output rather than taking part in the stretch.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_LOWER_PERCENTILE",
    "DEFAULT_UPPER_PERCENTILE",
    "percentile_limits",
    "stretch_to_uint8",
]

DEFAULT_LOWER_PERCENTILE = 2.0
DEFAULT_UPPER_PERCENTILE = 98.0

_PERCENTILE_MAX = 100.0
_UINT8_MAX = 255.0
_EXPECTED_BAND_CUBE_NDIM = 3
_EXPECTED_SINGLE_BAND_NDIM = 2


def percentile_limits(
    samples: npt.NDArray[Any],
    *,
    lower: float = DEFAULT_LOWER_PERCENTILE,
    upper: float = DEFAULT_UPPER_PERCENTILE,
) -> tuple[float, float]:
    """Return the stretch limits of a sample set.

    Args:
        samples: Values to summarise; non-finite entries are ignored.
        lower: Lower percentile in ``[0, 100)``.
        upper: Upper percentile in ``(lower, 100]``.

    Returns:
        ``(low, high)`` in the same units as ``samples``. When every sample is identical,
        ``high`` is ``low + 1`` so the subsequent division is defined and the band maps to
        a single grey level rather than raising.

    Raises:
        ValueError: If the percentile bounds are not ordered, or no finite sample remains.
    """
    if not (0.0 <= lower < upper <= _PERCENTILE_MAX):
        raise ValueError(
            f"percentiles must satisfy 0 <= lower < upper <= {_PERCENTILE_MAX:g}; "
            f"got lower={lower}, upper={upper}"
        )
    flat = np.asarray(samples, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("cannot compute percentiles: no finite samples remain")
    low, high = (float(value) for value in np.percentile(finite, [lower, upper]))
    if high <= low:
        high = low + 1.0
    return low, high


def _valid_samples(
    band: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[Any]:
    """Return the finite, unmasked samples of one band."""
    values = np.asarray(band, dtype=np.float64)
    if mask is None:
        return values[np.isfinite(values)]
    valid = ~np.asarray(mask, dtype=bool) & np.isfinite(values)
    return values[valid]


def stretch_to_uint8(
    data: npt.NDArray[Any],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
    per_band: bool = True,
) -> npt.NDArray[np.uint8]:
    """Map a raster array onto ``uint8`` by percentile stretch.

    Args:
        data: ``(bands, rows, columns)`` or ``(rows, columns)`` array.
        mask: Boolean mask with the same shape as ``data``; ``True`` means NoData and is
            excluded from the percentiles and written as 0.
        lower_percentile: Lower end of the display range.
        upper_percentile: Upper end of the display range.
        per_band: When true (default), each band gets its own limits. When false, one
            shared pair of limits is computed across every valid sample, which preserves
            relative colour balance at the cost of per-band contrast.

    Returns:
        A ``uint8`` array with the same shape as ``data``.

    Raises:
        ValueError: If the array rank is unsupported, the mask shape disagrees, the
            percentile bounds are invalid, or a band has no valid samples.
    """
    array = np.asarray(data)
    if array.ndim == _EXPECTED_SINGLE_BAND_NDIM:
        array = array[np.newaxis, ...]
        mask_cube = None if mask is None else np.asarray(mask, dtype=bool)[np.newaxis, ...]
        squeezed = True
    elif array.ndim == _EXPECTED_BAND_CUBE_NDIM:
        mask_cube = None if mask is None else np.asarray(mask, dtype=bool)
        squeezed = False
    else:
        raise ValueError(f"data must be 2- or 3-dimensional, got shape {tuple(array.shape)}")

    if mask_cube is not None and mask_cube.shape != array.shape:
        raise ValueError(
            f"mask shape {tuple(mask_cube.shape)} does not match data shape {tuple(array.shape)}"
        )

    band_count = int(array.shape[0])
    if per_band:
        limits = [
            percentile_limits(
                _valid_samples(array[index], None if mask_cube is None else mask_cube[index]),
                lower=lower_percentile,
                upper=upper_percentile,
            )
            for index in range(band_count)
        ]
    else:
        if mask_cube is None:
            pooled = array[np.isfinite(array)]
        else:
            pooled = array[~mask_cube & np.isfinite(array)]
        shared = percentile_limits(pooled, lower=lower_percentile, upper=upper_percentile)
        limits = [shared] * band_count

    output = np.zeros(array.shape, dtype=np.uint8)
    for index, (low, high) in enumerate(limits):
        band = array[index].astype(np.float64, copy=False)
        scaled = (band - low) / (high - low) * _UINT8_MAX
        np.clip(scaled, 0.0, _UINT8_MAX, out=scaled)
        stretched = np.rint(scaled).astype(np.uint8)
        if mask_cube is not None:
            stretched = np.where(mask_cube[index], np.uint8(0), stretched)
        else:
            stretched = np.where(np.isfinite(band), stretched, np.uint8(0))
        output[index] = stretched

    if squeezed:
        return cast("npt.NDArray[np.uint8]", output[0])
    return output
