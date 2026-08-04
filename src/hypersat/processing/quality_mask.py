"""Quality-mask generation in sensor geometry.

Class codes and precedence are fixed in ``docs/quality-masks.md``. Thresholds are
configuration, not constants: the defaults match a 16-bit DN product and are not a
mission-validated saturation specification.

Morphology is optional, disabled by default, and applied only to defect classes so that
bad data can grow but never be silently erased into ``VALID``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from hypersat.analytics.bands import nearest_band
from hypersat.exceptions import InvalidWavelengthMetadataError, QualityMaskError
from hypersat.io.environment import ensure_usable_proj_data
from hypersat.io.files import derive_product_id
from hypersat.io.inspect import inspect_raster, resolve_raster_path
from hypersat.io.reader import ReadOptions, read_chunk
from hypersat.io.writer import write_array
from hypersat.logging_config import get_logger
from hypersat.models.config import (
    MorphologyConfig,
    MorphologyOperation,
    QualityMaskRequest,
)
from hypersat.models.product import RasterInfo
from hypersat.models.raster import RasterMetadata
from hypersat.processing.validation import validate_output_directory

__all__ = [
    "QualityClass",
    "QualityMaskResult",
    "apply_defect_morphology",
    "build_quality_mask",
    "class_counts",
    "classify_quality",
]

logger = get_logger(__name__)

_EXPECTED_CUBE_NDIM = 3


class QualityClass(IntEnum):
    """Single-class quality codes. ``NO_DATA`` is also the GeoTIFF NoData value."""

    NO_DATA = 0
    VALID = 1
    SATURATED = 2
    LOW_SIGNAL = 3
    INVALID_NUMERIC = 4
    SPECTRAL_ANOMALY = 5
    UNCLASSIFIED = 255


_DEFECT_PRECEDENCE = (
    QualityClass.INVALID_NUMERIC,
    QualityClass.SATURATED,
    QualityClass.LOW_SIGNAL,
    QualityClass.SPECTRAL_ANOMALY,
)


@dataclass(frozen=True, slots=True)
class QualityMaskResult:
    """Outcome of a successful quality-mask build."""

    path: Path
    band_indices: tuple[int, ...]
    wavelengths_nm: tuple[float | None, ...]
    counts: Mapping[str, int]
    product_id: str
    statistics_path: Path | None


def classify_quality(
    data: npt.NDArray[Any],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    nodata: float | None = None,
    saturation_dn: float,
    low_signal_dn: float,
    band_fraction: float,
    spectral_anomaly: bool = False,
    anomaly_cv_threshold: float = 2.0,
) -> npt.NDArray[np.uint8]:
    """Classify a band cube into a single-band quality mask.

    Args:
        data: Array shaped ``(bands, rows, columns)`` of the evaluation bands.
        mask: Optional boolean mask with the same shape; ``True`` means source NoData.
        nodata: Optional scalar NoData value to treat like a mask.
        saturation_dn: DN at or above this is saturated.
        low_signal_dn: DN at or below this is low signal.
        band_fraction: Fraction of *valid* evaluation bands that must meet a threshold.
        spectral_anomaly: When true, flag high coefficient-of-variation spectra.
        anomaly_cv_threshold: CV above which a valid spectrum is anomalous.

    Returns:
        ``uint8`` class raster shaped ``(rows, columns)``.

    Raises:
        ValueError: If shapes disagree or ``band_fraction`` is outside ``(0, 1]``.
    """
    cube = np.asarray(data, dtype=np.float64)
    if cube.ndim != _EXPECTED_CUBE_NDIM:
        raise ValueError(f"data must be (bands, rows, columns), got shape {cube.shape}")
    if not (0.0 < band_fraction <= 1.0):
        raise ValueError(f"band_fraction must be in (0, 1], got {band_fraction}")

    _band_count, height, width = cube.shape
    if mask is not None:
        mask_cube = np.asarray(mask, dtype=bool)
        if mask_cube.shape != cube.shape:
            raise ValueError(f"mask shape {mask_cube.shape} does not match data shape {cube.shape}")
    else:
        mask_cube = np.zeros(cube.shape, dtype=bool)

    if nodata is not None and np.isfinite(nodata):
        mask_cube = mask_cube | (cube == nodata)

    finite = np.isfinite(cube) & ~mask_cube
    valid_band_count = finite.sum(axis=0).astype(np.float64)
    all_nodata = valid_band_count == 0
    any_invalid = (~np.isfinite(cube) & ~mask_cube).any(axis=0)

    saturated_count = ((cube >= saturation_dn) & finite).sum(axis=0).astype(np.float64)
    low_count = ((cube <= low_signal_dn) & finite).sum(axis=0).astype(np.float64)
    # Avoid divide-by-zero on all-nodata pixels; those are already NO_DATA.
    safe_count = np.maximum(valid_band_count, 1.0)
    saturated = (saturated_count / safe_count) >= band_fraction
    low_signal = (low_count / safe_count) >= band_fraction

    anomaly = np.zeros((height, width), dtype=bool)
    if spectral_anomaly:
        # Coefficient of variation across evaluation bands; a rough plausibility check,
        # not a cloud or land-cover classifier.
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(np.where(finite, cube, np.nan), axis=0)
            std = np.nanstd(np.where(finite, cube, np.nan), axis=0)
            cv = np.where(mean > 0.0, std / mean, np.inf)
        anomaly = np.isfinite(cv) & (cv > anomaly_cv_threshold) & ~all_nodata

    result = np.full((height, width), QualityClass.VALID, dtype=np.uint8)
    # Precedence: most severe first (later assignments must not overwrite earlier ones).
    # Build from least to most severe so final writes win, or set only where still VALID.
    result[anomaly] = QualityClass.SPECTRAL_ANOMALY
    result[low_signal] = QualityClass.LOW_SIGNAL
    result[saturated] = QualityClass.SATURATED
    result[any_invalid] = QualityClass.INVALID_NUMERIC
    result[all_nodata] = QualityClass.NO_DATA
    return result


def _structuring_element(kernel_shape: str, kernel_size: int) -> npt.NDArray[np.uint8]:
    """Build an OpenCV structuring element; kernel size must be a positive odd integer."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise QualityMaskError(
            "Morphology kernel_size must be a positive odd integer.",
            hint="Pass an odd value such as 3, 5 or 7.",
            context={"kernel_size": kernel_size},
        )
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    try:
        shape = shape_map[kernel_shape]
    except KeyError as error:
        raise QualityMaskError(
            f"Unknown morphology kernel_shape {kernel_shape!r}.",
            hint="Use rect, ellipse or cross.",
            context={"kernel_shape": kernel_shape},
        ) from error
    return cast(
        "npt.NDArray[np.uint8]",
        cv2.getStructuringElement(shape, (kernel_size, kernel_size)),
    )


def apply_defect_morphology(
    class_mask: npt.NDArray[np.uint8],
    config: MorphologyConfig,
) -> npt.NDArray[np.uint8]:
    """Grow defect classes with OpenCV morphology; never reclassify into ``VALID``.

    Each defect class is processed from most severe to least. Morphology may only paint
    over ``VALID`` pixels, so ``NO_DATA`` and more severe defects are preserved.

    Args:
        class_mask: Single-band class raster.
        config: Morphology options; a no-op when ``enabled`` is false.

    Returns:
        A new class raster.
    """
    if not config.enabled or config.operation is MorphologyOperation.NONE:
        return class_mask

    result = class_mask.copy()
    kernel = _structuring_element(config.kernel_shape.value, config.kernel_size)
    op_map = {
        MorphologyOperation.OPEN: cv2.MORPH_OPEN,
        MorphologyOperation.CLOSE: cv2.MORPH_CLOSE,
        MorphologyOperation.DILATE: cv2.MORPH_DILATE,
        MorphologyOperation.ERODE: cv2.MORPH_ERODE,
    }
    morph_op = op_map[config.operation]

    for class_code in _DEFECT_PRECEDENCE:
        binary = (class_mask == int(class_code)).astype(np.uint8)
        if not np.any(binary):
            continue
        transformed = cv2.morphologyEx(binary, morph_op, kernel, iterations=config.iterations)
        grow = (transformed == 1) & (result == int(QualityClass.VALID))
        result[grow] = int(class_code)
    return result


def class_counts(class_mask: npt.NDArray[np.uint8]) -> dict[str, int]:
    """Return pixel counts keyed by class name."""
    counts = {member.name.lower(): 0 for member in QualityClass}
    unique, frequencies = np.unique(class_mask, return_counts=True)
    for code, frequency in zip(unique.tolist(), frequencies.tolist(), strict=True):
        try:
            name = QualityClass(code).name.lower()
        except ValueError:
            name = f"unknown_{code}"
            counts[name] = 0
        counts[name] = int(frequency)
    return counts


def _resolve_evaluation_bands(
    info: RasterInfo,
    *,
    wavelengths_nm: Sequence[float] | None,
    bands: Sequence[int] | None,
    tolerance_nm: float,
) -> tuple[tuple[int, ...], tuple[float | None, ...]]:
    """Resolve evaluation bands from explicit indices or target wavelengths."""
    if bands is not None:
        out_of_range = sorted(index for index in bands if index < 1 or index > info.band_count)
        if out_of_range:
            raise QualityMaskError(
                "Requested evaluation bands are outside the raster's band range.",
                hint=f"This raster has {info.band_count} band(s); indices are 1-based.",
                context={"out_of_range": out_of_range, "band_count": info.band_count},
            )
        indices = tuple(bands)
        return indices, tuple(info.bands[index - 1].wavelength_nm for index in indices)

    if wavelengths_nm is None:
        indices = tuple(range(1, info.band_count + 1))
        return indices, tuple(band.wavelength_nm for band in info.bands)

    product_wavelengths = [band.wavelength_nm for band in info.bands]
    try:
        matches = [
            nearest_band(product_wavelengths, target, tolerance_nm=tolerance_nm)
            for target in wavelengths_nm
        ]
    except InvalidWavelengthMetadataError as error:
        raise QualityMaskError(
            error.message,
            hint=error.hint
            or "Pass --bands with explicit 1-based indices, or provide wavelength metadata.",
            context=error.context,
        ) from error

    # Preserve order but drop duplicate band hits from neighbouring wavelength targets.
    seen: set[int] = set()
    indices_list: list[int] = []
    wavelengths_list: list[float | None] = []
    for match in matches:
        if match.index in seen:
            continue
        seen.add(match.index)
        indices_list.append(match.index)
        wavelengths_list.append(match.wavelength_nm)
    return tuple(indices_list), tuple(wavelengths_list)


def build_quality_mask(request: QualityMaskRequest) -> QualityMaskResult:
    """Build a uint8 quality-mask GeoTIFF in the source geometry.

    Args:
        request: Validated quality-mask configuration.

    Returns:
        Paths and class-count summary for the written mask.

    Raises:
        QualityMaskError: If band selection or classification fails.
        RasterReadError: If the raster cannot be read.
        RasterWriteError: If the mask cannot be written.
        OutputPathError: If the output directory is unusable.
    """
    ensure_usable_proj_data(allow_repair=request.proj_autofix)
    validate_output_directory(request.output.directory, overwrite=request.output.overwrite)

    raster_path, _layout = resolve_raster_path(request.product_path)
    info = inspect_raster(raster_path)
    resolved_id = request.product_id or derive_product_id(request.product_path)
    band_indices, wavelengths = _resolve_evaluation_bands(
        info,
        wavelengths_nm=request.evaluation_wavelengths_nm,
        bands=request.bands,
        tolerance_nm=request.tolerance_nm,
    )
    if not band_indices:
        raise QualityMaskError(
            "No evaluation bands were selected for the quality mask.",
            hint="Pass --bands or --evaluation-wavelengths-nm with at least one entry.",
            context={"band_count": info.band_count},
        )

    chunk = read_chunk(
        raster_path,
        bands=band_indices,
        options=ReadOptions(masked=True, as_float32=True),
    )
    classified = classify_quality(
        chunk.data,
        mask=chunk.mask,
        nodata=chunk.metadata.nodata,
        saturation_dn=request.saturation_dn,
        low_signal_dn=request.low_signal_dn,
        band_fraction=request.saturation_band_fraction,
        spectral_anomaly=request.spectral_anomaly,
        anomaly_cv_threshold=request.anomaly_cv_threshold,
    )
    classified = apply_defect_morphology(classified, request.morphology)

    destination = request.output.directory / f"{resolved_id}_qmask.tif"
    metadata = RasterMetadata(
        crs_wkt=chunk.metadata.crs_wkt,
        transform=chunk.metadata.transform,
        nodata=float(QualityClass.NO_DATA),
        band_descriptions=("quality_mask",),
        dataset_tags={
            "QUALITY_MASK_VERSION": "1",
            "EVALUATION_BANDS": ",".join(str(index) for index in band_indices),
            "SATURATION_DN": f"{request.saturation_dn:g}",
            "LOW_SIGNAL_DN": f"{request.low_signal_dn:g}",
            "BAND_FRACTION": f"{request.saturation_band_fraction:g}",
            "MORPHOLOGY": (
                "disabled"
                if not request.morphology.enabled
                else (
                    f"{request.morphology.operation.value}/"
                    f"{request.morphology.kernel_shape.value}/"
                    f"{request.morphology.kernel_size}/"
                    f"{request.morphology.iterations}"
                )
            ),
        },
        rpc_tags=chunk.metadata.rpc_tags if chunk.covers_full_grid else None,
    )
    write_array(
        destination,
        classified,
        metadata=metadata,
        overwrite=request.output.overwrite,
    )

    counts = class_counts(classified)
    stats_path: Path | None = None
    if request.include_statistics:
        stats_path = request.output.directory / f"{resolved_id}_qmask_statistics.json"
        if stats_path.exists() and not request.output.overwrite:
            raise QualityMaskError(
                "The quality-mask statistics file already exists.",
                hint="Pass --overwrite, or choose a different --output-dir / product id.",
                context={"path": str(stats_path)},
            )
        total = int(classified.size)
        percentages = {
            name: (0.0 if total == 0 else 100.0 * count / total) for name, count in counts.items()
        }
        stats_path.write_text(
            json.dumps(
                {
                    "path": str(destination),
                    "band_indices": list(band_indices),
                    "wavelengths_nm": list(wavelengths),
                    "counts": dict(counts),
                    "percentages": percentages,
                    "valid_percentage": percentages.get("valid", 0.0),
                    "nodata_percentage": percentages.get("no_data", 0.0),
                    "saturated_percentage": percentages.get("saturated", 0.0),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    logger.info(
        "wrote quality mask",
        extra={
            "path": str(destination),
            "bands": list(band_indices),
            "valid": counts.get("valid", 0),
            "nodata": counts.get("no_data", 0),
            "saturated": counts.get("saturated", 0),
        },
    )
    return QualityMaskResult(
        path=destination,
        band_indices=band_indices,
        wavelengths_nm=wavelengths,
        counts=counts,
        product_id=resolved_id,
        statistics_path=stats_path,
    )
