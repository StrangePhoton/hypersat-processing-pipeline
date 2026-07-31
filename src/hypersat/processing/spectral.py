"""I/O orchestration for spectral indices and profiles.

Pure arithmetic lives in :mod:`hypersat.analytics`. This module opens rasters, selects
bands by wavelength, writes GeoTIFF / CSV / JSON products and never invents calibration.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from hypersat.analytics.bands import nearest_band
from hypersat.analytics.indices import ndvi, ndwi
from hypersat.analytics.profiles import SpectralProfile, extract_profile
from hypersat.analytics.statistics import BandStatistics, summarise_array
from hypersat.exceptions import InvalidWavelengthMetadataError, SpectralAnalysisError
from hypersat.io.environment import ensure_usable_proj_data
from hypersat.io.files import derive_product_id
from hypersat.io.inspect import inspect_raster, resolve_raster_path
from hypersat.io.reader import ReadOptions, read_chunk
from hypersat.io.writer import write_array
from hypersat.logging_config import get_logger
from hypersat.models.config import IndexRequest, SpectralIndexName, SpectralProfileRequest
from hypersat.models.product import RasterInfo
from hypersat.models.raster import RasterMetadata, ReadWindow
from hypersat.processing.validation import validate_output_directory

__all__ = [
    "IndexResult",
    "ProfileResult",
    "calculate_index",
    "extract_spectral_profile",
]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Outcome of a successful index calculation."""

    path: Path
    index: SpectralIndexName
    band_indices: tuple[int, int]
    wavelengths_nm: tuple[float, float]
    nodata: float
    product_id: str
    statistics: BandStatistics | None
    statistics_path: Path | None


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Outcome of a successful spectral-profile extraction."""

    profile: SpectralProfile
    csv_path: Path
    json_path: Path
    product_id: str


def _resolve_index_bands(
    info: RasterInfo,
    request: IndexRequest,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Return ``((minuend_index, subtrahend_index), (minuend_nm, subtrahend_nm))``."""
    if request.bands is not None:
        minuend_index, subtrahend_index = request.bands
        if max(minuend_index, subtrahend_index) > info.band_count:
            raise SpectralAnalysisError(
                "Requested band indices are outside the raster's band range.",
                hint=f"This raster has {info.band_count} band(s); indices are 1-based.",
                context={
                    "bands": list(request.bands),
                    "band_count": info.band_count,
                },
            )
        minuend_wavelength = info.bands[minuend_index - 1].wavelength_nm
        subtrahend_wavelength = info.bands[subtrahend_index - 1].wavelength_nm
        return (minuend_index, subtrahend_index), (
            float("nan") if minuend_wavelength is None else minuend_wavelength,
            float("nan") if subtrahend_wavelength is None else subtrahend_wavelength,
        )

    wavelengths_nm = [band.wavelength_nm for band in info.bands]
    try:
        if request.index is SpectralIndexName.NDVI:
            minuend = nearest_band(
                wavelengths_nm, request.nir_nm, tolerance_nm=request.tolerance_nm
            )
            subtrahend = nearest_band(
                wavelengths_nm, request.red_nm, tolerance_nm=request.tolerance_nm
            )
        else:
            minuend = nearest_band(
                wavelengths_nm, request.green_nm, tolerance_nm=request.tolerance_nm
            )
            subtrahend = nearest_band(
                wavelengths_nm, request.nir_nm, tolerance_nm=request.tolerance_nm
            )
    except InvalidWavelengthMetadataError as error:
        raise SpectralAnalysisError(
            error.message,
            hint=error.hint
            or "Pass --bands with two 1-based indices to override wavelength selection.",
            context=error.context,
        ) from error

    if minuend.index == subtrahend.index:
        raise SpectralAnalysisError(
            "Both sides of the index resolved to the same band.",
            hint="Widen the spectral separation, supply --bands, or check wavelength metadata.",
            context={
                "index": request.index.value,
                "band": minuend.index,
                "wavelength_nm": minuend.wavelength_nm,
            },
        )
    return (
        (minuend.index, subtrahend.index),
        (minuend.wavelength_nm, subtrahend.wavelength_nm),
    )


def calculate_index(request: IndexRequest) -> IndexResult:
    """Compute one spectral index and write it as a float32 GeoTIFF.

    Args:
        request: Validated index configuration.

    Returns:
        Paths and band metadata for the written product.

    Raises:
        SpectralAnalysisError: If band selection or the arithmetic cannot proceed.
        RasterReadError: If the raster cannot be read.
        RasterWriteError: If the GeoTIFF cannot be written.
        OutputPathError: If the output directory is unusable.
    """
    ensure_usable_proj_data(allow_repair=request.proj_autofix)
    validate_output_directory(request.output.directory, overwrite=request.output.overwrite)

    raster_path, _layout = resolve_raster_path(request.product_path)
    info = inspect_raster(raster_path)
    resolved_id = request.product_id or derive_product_id(request.product_path)
    band_pair, wavelengths = _resolve_index_bands(info, request)

    chunk = read_chunk(
        raster_path,
        bands=band_pair,
        options=ReadOptions(masked=True, as_float32=True),
    )
    combined_mask = chunk.mask[0] | chunk.mask[1]
    if request.index is SpectralIndexName.NDVI:
        values = ndvi(
            chunk.data[0],
            chunk.data[1],
            mask=combined_mask,
            nodata=request.output_nodata,
        )
        description = (
            f"NDVI from bands {band_pair[0]} ({wavelengths[0]:g} nm) and "
            f"{band_pair[1]} ({wavelengths[1]:g} nm)"
        )
        formula = "(NIR-Red)/(NIR+Red)"
    else:
        values = ndwi(
            chunk.data[0],
            chunk.data[1],
            mask=combined_mask,
            nodata=request.output_nodata,
        )
        description = (
            f"NDWI (McFeeters) from bands {band_pair[0]} ({wavelengths[0]:g} nm) and "
            f"{band_pair[1]} ({wavelengths[1]:g} nm)"
        )
        formula = "(Green-NIR)/(Green+NIR)"

    destination = request.output.directory / f"{resolved_id}_{request.index.value}.tif"
    metadata = RasterMetadata(
        crs_wkt=chunk.metadata.crs_wkt,
        transform=chunk.metadata.transform,
        nodata=request.output_nodata,
        band_descriptions=(description,),
        dataset_tags={
            "INDEX": request.index.value,
            "INDEX_BANDS": f"{band_pair[0]},{band_pair[1]}",
            "INDEX_WAVELENGTHS_NM": f"{wavelengths[0]:g},{wavelengths[1]:g}",
            "INDEX_FORMULA": formula,
        },
    )
    write_array(
        destination,
        values,
        metadata=metadata,
        overwrite=request.output.overwrite,
    )

    stats: BandStatistics | None = None
    stats_path: Path | None = None
    if request.include_statistics:
        stats = summarise_array(
            values,
            nodata=request.output_nodata,
            band_index=1,
            sample_step=request.statistics_sample_step,
        )
        stats_path = (
            request.output.directory / f"{resolved_id}_{request.index.value}_statistics.json"
        )
        if stats_path.exists() and not request.output.overwrite:
            raise SpectralAnalysisError(
                "The statistics file already exists.",
                hint="Pass --overwrite, or choose a different --output-dir / product id.",
                context={"path": str(stats_path)},
            )
        stats_path.write_text(
            json.dumps(
                {
                    "index": request.index.value,
                    "path": str(destination),
                    "band_indices": list(band_pair),
                    "wavelengths_nm": list(wavelengths),
                    "nodata": request.output_nodata,
                    "statistics": stats.to_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    logger.info(
        "wrote spectral index",
        extra={
            "path": str(destination),
            "index": request.index.value,
            "bands": list(band_pair),
        },
    )
    return IndexResult(
        path=destination,
        index=request.index,
        band_indices=band_pair,
        wavelengths_nm=wavelengths,
        nodata=request.output_nodata,
        product_id=resolved_id,
        statistics=stats,
        statistics_path=stats_path,
    )


def extract_spectral_profile(request: SpectralProfileRequest) -> ProfileResult:
    """Extract a pixel (or neighbourhood) spectrum and write CSV + JSON.

    Args:
        request: Validated profile configuration.

    Returns:
        The profile and the paths written.

    Raises:
        SpectralAnalysisError: If the pixel falls outside the raster or bands are invalid.
        RasterReadError: If the raster cannot be read.
        OutputPathError: If the output directory is unusable.
    """
    ensure_usable_proj_data(allow_repair=request.proj_autofix)
    validate_output_directory(request.output.directory, overwrite=request.output.overwrite)

    raster_path, _layout = resolve_raster_path(request.product_path)
    info = inspect_raster(raster_path)
    resolved_id = request.product_id or derive_product_id(request.product_path)

    if request.row >= info.height or request.col >= info.width:
        raise SpectralAnalysisError(
            "Requested pixel lies outside the raster.",
            hint=f"This raster is {info.width} x {info.height} pixels; row/col are 0-based.",
            context={
                "row": request.row,
                "col": request.col,
                "width": info.width,
                "height": info.height,
            },
        )

    if request.bands is None:
        band_indices = tuple(range(1, info.band_count + 1))
    else:
        out_of_range = sorted(
            index for index in request.bands if index < 1 or index > info.band_count
        )
        if out_of_range:
            raise SpectralAnalysisError(
                "Requested band indices are outside the raster's band range.",
                hint=f"This raster has {info.band_count} band(s); indices are 1-based.",
                context={"out_of_range": out_of_range, "band_count": info.band_count},
            )
        band_indices = request.bands

    # Read only the neighbourhood window so a 224-band profile stays cheap.
    half = request.window_size // 2
    row_off = max(0, request.row - half)
    col_off = max(0, request.col - half)
    row_end = min(info.height, request.row + half + 1)
    col_end = min(info.width, request.col + half + 1)
    window = ReadWindow(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
    )
    chunk = read_chunk(
        raster_path,
        window=window,
        bands=band_indices,
        options=ReadOptions(masked=True, as_float32=True),
    )

    local_row = request.row - row_off
    local_col = request.col - col_off
    wavelengths = tuple(info.bands[index - 1].wavelength_nm for index in band_indices)
    local_profile = extract_profile(
        chunk.data,
        local_row,
        local_col,
        band_indices=band_indices,
        wavelengths_nm=wavelengths,
        mask=chunk.mask,
        window_size=request.window_size,
    )
    profile = SpectralProfile(
        row=request.row,
        col=request.col,
        window_size=local_profile.window_size,
        values=local_profile.values,
        band_indices=local_profile.band_indices,
        wavelengths_nm=local_profile.wavelengths_nm,
    )

    stem = f"{resolved_id}_spectral_profile_r{request.row}_c{request.col}"
    csv_path = request.output.directory / f"{stem}.csv"
    json_path = request.output.directory / f"{stem}.json"
    for path in (csv_path, json_path):
        if path.exists() and not request.output.overwrite:
            raise SpectralAnalysisError(
                "The spectral-profile output already exists.",
                hint="Pass --overwrite, or choose a different --output-dir / product id.",
                context={"path": str(path)},
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band_index", "wavelength_nm", "value"])
        for band_index, wavelength, value in zip(
            profile.band_indices,
            profile.wavelengths_nm,
            profile.values,
            strict=True,
        ):
            writer.writerow(
                [
                    band_index,
                    "" if wavelength is None else f"{wavelength:g}",
                    "" if value is None else f"{value:.8g}",
                ]
            )

    payload = {
        "product_id": resolved_id,
        "row": profile.row,
        "col": profile.col,
        "window_size": profile.window_size,
        "band_indices": list(profile.band_indices),
        "wavelengths_nm": list(profile.wavelengths_nm),
        "values": list(profile.values),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "wrote spectral profile",
        extra={
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "row": request.row,
            "col": request.col,
            "bands": len(band_indices),
        },
    )
    return ProfileResult(
        profile=profile,
        csv_path=csv_path,
        json_path=json_path,
        product_id=resolved_id,
    )
