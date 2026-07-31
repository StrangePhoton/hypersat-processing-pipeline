"""Pixel spectral-profile extraction as pure array functions.

A profile is the vector of values at one pixel (or the mean of a small neighbourhood)
across the selected bands. Coordinates are zero-based row/column indices into the array
the caller already holds - map coordinates are a later concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "SpectralProfile",
    "extract_profile",
]

_EXPECTED_CUBE_NDIM = 3


@dataclass(frozen=True, slots=True)
class SpectralProfile:
    """Values extracted at one pixel (or neighbourhood) across bands.

    Attributes:
        row: Zero-based centre row in the source array.
        col: Zero-based centre column in the source array.
        window_size: Odd neighbourhood edge length used; ``1`` is a single pixel.
        values: One value per band; ``None`` where every sample in the neighbourhood was
            masked or non-finite.
        band_indices: One-based source band indices matching ``values``.
        wavelengths_nm: Centre wavelengths matching ``values``, when known.
    """

    row: int
    col: int
    window_size: int
    values: tuple[float | None, ...]
    band_indices: tuple[int, ...]
    wavelengths_nm: tuple[float | None, ...]


def extract_profile(
    data: npt.NDArray[Any],
    row: int,
    col: int,
    *,
    band_indices: tuple[int, ...],
    wavelengths_nm: tuple[float | None, ...] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    window_size: int = 1,
) -> SpectralProfile:
    """Extract the spectrum at ``(row, col)``, optionally averaging a neighbourhood.

    Args:
        data: Array shaped ``(bands, rows, columns)``.
        row: Zero-based centre row.
        col: Zero-based centre column.
        band_indices: One-based band indices corresponding to the leading axis of ``data``.
        wavelengths_nm: Optional per-band centre wavelengths.
        mask: Optional boolean mask with the same shape as ``data``; ``True`` is NoData.
        window_size: Odd positive neighbourhood size. ``1`` reads a single pixel; larger
            windows average the finite, unmasked samples around the centre.

    Returns:
        The extracted profile.

    Raises:
        ValueError: If shapes disagree, the window is not a positive odd integer, or the
            centre falls outside the array.
    """
    cube = np.asarray(data)
    if cube.ndim != _EXPECTED_CUBE_NDIM:
        raise ValueError(f"data must be (bands, rows, columns), got shape {cube.shape}")
    band_count, height, width = cube.shape
    if band_count != len(band_indices):
        raise ValueError(
            f"data has {band_count} band(s) but {len(band_indices)} band index/indices were given"
        )
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"pixel ({row}, {col}) lies outside the array of size {height} x {width}")
    if mask is not None and np.asarray(mask).shape != cube.shape:
        raise ValueError(
            f"mask shape {np.asarray(mask).shape} does not match data shape {cube.shape}"
        )

    half = window_size // 2
    row_start = max(0, row - half)
    row_end = min(height, row + half + 1)
    col_start = max(0, col - half)
    col_end = min(width, col + half + 1)

    values: list[float | None] = []
    for band_position in range(band_count):
        patch = cube[band_position, row_start:row_end, col_start:col_end].astype(
            np.float64, copy=False
        )
        if mask is None:
            valid = np.isfinite(patch)
        else:
            valid = ~np.asarray(mask[band_position, row_start:row_end, col_start:col_end])
            valid &= np.isfinite(patch)
        if not np.any(valid):
            values.append(None)
        else:
            values.append(float(np.mean(patch[valid])))

    wavelengths = wavelengths_nm if wavelengths_nm is not None else (None,) * band_count
    if len(wavelengths) != band_count:
        raise ValueError(
            f"wavelengths_nm has length {len(wavelengths)} but data has {band_count} band(s)"
        )

    return SpectralProfile(
        row=row,
        col=col,
        window_size=window_size,
        values=tuple(values),
        band_indices=band_indices,
        wavelengths_nm=tuple(wavelengths),
    )
