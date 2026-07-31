"""Per-band descriptive statistics as pure array functions.

Statistics honour NoData and non-finite values. On a large cube, ``sample_step`` keeps the
cost predictable by taking every Nth pixel along each axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "BandStatistics",
    "band_statistics",
    "summarise_array",
]

_EXPECTED_SINGLE_BAND_NDIM = 2
_EXPECTED_CUBE_NDIM = 3


@dataclass(frozen=True, slots=True)
class BandStatistics:
    """Descriptive statistics for one band.

    Attributes:
        band_index: One-based source band index, when known.
        valid_count: Number of finite, unmasked samples that contributed.
        minimum: Minimum valid value, or ``None`` when ``valid_count`` is zero.
        maximum: Maximum valid value, or ``None`` when ``valid_count`` is zero.
        mean: Arithmetic mean of valid values, or ``None``.
        std: Population standard deviation of valid values, or ``None``.
        percentile_2: 2nd percentile, or ``None``.
        percentile_98: 98th percentile, or ``None``.
    """

    band_index: int | None
    valid_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    percentile_2: float | None
    percentile_98: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a JSON-serialisable representation."""
        return {
            "band_index": self.band_index,
            "valid_count": self.valid_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "percentile_2": self.percentile_2,
            "percentile_98": self.percentile_98,
        }


def summarise_array(
    values: npt.NDArray[Any],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    nodata: float | None = None,
    band_index: int | None = None,
    sample_step: int = 1,
) -> BandStatistics:
    """Summarise one band (or any 2-D / 1-D array of samples).

    Args:
        values: Sample array.
        mask: Optional boolean mask; ``True`` means NoData.
        nodata: Optional scalar NoData value to exclude in addition to the mask.
        band_index: Optional 1-based band index recorded in the result.
        sample_step: Take every Nth sample along each axis; ``1`` uses every sample.

    Returns:
        Statistics over the surviving samples.

    Raises:
        ValueError: If ``sample_step`` is not positive, or the mask shape disagrees.
    """
    if sample_step < 1:
        raise ValueError(f"sample_step must be positive, got {sample_step}")

    array = np.asarray(values, dtype=np.float64)
    if sample_step > 1:
        if array.ndim == 0:
            sampled = array
        elif array.ndim == 1:
            sampled = array[::sample_step]
        else:
            slicer = (slice(None, None, sample_step),) * array.ndim
            sampled = array[slicer]
    else:
        sampled = array

    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if sample_step > 1 and mask_array.ndim > 0:
            slicer = (slice(None, None, sample_step),) * mask_array.ndim
            mask_array = mask_array[slicer]
        if mask_array.shape != sampled.shape:
            raise ValueError(
                f"mask shape {mask_array.shape} does not match values shape {sampled.shape}"
            )
        valid = ~mask_array & np.isfinite(sampled)
    else:
        valid = np.isfinite(sampled)

    if nodata is not None and np.isfinite(nodata):
        valid &= sampled != nodata

    finite = sampled[valid]
    count = int(finite.size)
    if count == 0:
        return BandStatistics(
            band_index=band_index,
            valid_count=0,
            minimum=None,
            maximum=None,
            mean=None,
            std=None,
            percentile_2=None,
            percentile_98=None,
        )

    percentiles = np.percentile(finite, [2.0, 98.0])
    return BandStatistics(
        band_index=band_index,
        valid_count=count,
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        percentile_2=float(percentiles[0]),
        percentile_98=float(percentiles[1]),
    )


def band_statistics(
    data: npt.NDArray[Any],
    *,
    band_indices: tuple[int, ...] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    nodata: float | None = None,
    sample_step: int = 1,
) -> tuple[BandStatistics, ...]:
    """Compute :class:`BandStatistics` for every band of a cube.

    Args:
        data: ``(bands, rows, columns)`` array, or a single 2-D band.
        band_indices: Optional 1-based indices matching the leading axis.
        mask: Optional mask with the same shape as ``data``.
        nodata: Optional scalar NoData value.
        sample_step: Subsampling stride along each spatial axis.

    Returns:
        One :class:`BandStatistics` per band.
    """
    array = np.asarray(data)
    if array.ndim == _EXPECTED_SINGLE_BAND_NDIM:
        array = array[np.newaxis, ...]
        mask_cube = None if mask is None else np.asarray(mask)[np.newaxis, ...]
    elif array.ndim == _EXPECTED_CUBE_NDIM:
        mask_cube = None if mask is None else np.asarray(mask)
    else:
        raise ValueError(f"data must be 2- or 3-dimensional, got shape {array.shape}")

    count = int(array.shape[0])
    indices = band_indices if band_indices is not None else tuple(range(1, count + 1))
    if len(indices) != count:
        raise ValueError(f"band_indices has length {len(indices)} but data has {count} band(s)")

    return tuple(
        summarise_array(
            array[position],
            mask=None if mask_cube is None else mask_cube[position],
            nodata=nodata,
            band_index=indices[position],
            sample_step=sample_step,
        )
        for position in range(count)
    )
