"""RPC + DEM orthorectification.

Warps sensor-geometry imagery onto a map grid using the Rational Polynomial
Coefficients sensor model and a digital elevation model. Plain reprojection is never
used as a fallback: without both inputs, relief displacement cannot be corrected
(see ``docs/orthorectification.md``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError, RasterioIOError, WarpOperationError
from rasterio.rpc import RPC
from rasterio.warp import (
    aligned_target,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

from hypersat.exceptions import (
    MemoryBudgetExceededError,
    MissingRPCMetadataError,
    OrthorectificationError,
    UnreadableDEMError,
)
from hypersat.io.environment import ensure_usable_proj_data
from hypersat.io.files import derive_product_id
from hypersat.io.inspect import inspect_raster, resolve_raster_path
from hypersat.io.reader import DEFAULT_READ_BUDGET_BYTES
from hypersat.io.writer import write_array
from hypersat.logging_config import get_logger
from hypersat.models.config import (
    DataSemantics,
    OrthorectifyRequest,
    ResamplingMethod,
    RpcTransformerOptions,
)
from hypersat.models.product import RasterInfo, RPCInfo
from hypersat.models.raster import RasterMetadata
from hypersat.processing.reprojection import (
    OutputGrid,
    format_resolution_token,
    resampling_for_semantics,
    resolve_target_crs,
)
from hypersat.processing.validation import validate_dem, validate_output_directory

__all__ = [
    "OrthorectifyResult",
    "orthorectify_raster",
    "rpc_geographic_bounds",
    "validate_dem_covers_scene",
]

logger = get_logger(__name__)

WGS84_EPSG = 4326
BOUNDS_DENSIFY_POINTS = 21


@dataclass(frozen=True, slots=True)
class OrthorectifyResult:
    """Outcome of a successful orthorectification."""

    path: Path
    crs_authority: str
    resolution: float
    width: int
    height: int
    band_indices: tuple[int, ...]
    resampling: ResamplingMethod
    dem_path: Path
    transformer_options: Mapping[str, str]
    product_id: str


def rpc_geographic_bounds(rpc: RPCInfo) -> tuple[float, float, float, float]:
    """Approximate scene footprint in WGS84 from RPC normalisation offsets/scales.

    Returns:
        ``(left, bottom, right, top)`` longitude/latitude degrees covering the RPC
        normalisation domain.

    Raises:
        OrthorectificationError: If required RPC scalars are missing.
    """
    required = {
        "long_off": rpc.long_off,
        "lat_off": rpc.lat_off,
        "long_scale": rpc.long_scale,
        "lat_scale": rpc.lat_scale,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise OrthorectificationError(
            "RPC metadata is missing normalisation scalars needed for the scene footprint.",
            hint="Re-download the product; a usable RPC00B model must include LONG/LAT "
            "offset and scale terms.",
            context={"missing": missing},
        )
    long_off = cast("float", rpc.long_off)
    lat_off = cast("float", rpc.lat_off)
    long_scale = abs(cast("float", rpc.long_scale))
    lat_scale = abs(cast("float", rpc.lat_scale))
    return (
        long_off - long_scale,
        lat_off - lat_scale,
        long_off + long_scale,
        lat_off + lat_scale,
    )


def validate_dem_covers_scene(
    dem_info: RasterInfo,
    scene_bounds_wgs84: tuple[float, float, float, float],
) -> None:
    """Require the DEM footprint to overlap the scene's geographic domain.

    Running an RPC warp with a DEM that misses the scene silently degrades to a
    constant-height approximation, which this project refuses.

    Raises:
        UnreadableDEMError: If the DEM CRS/bounds cannot be interpreted or do not overlap.
    """
    if dem_info.bounds is None or dem_info.crs.wkt is None:
        raise UnreadableDEMError(
            "DEM bounds are unavailable, so scene coverage cannot be verified.",
            hint="Use a georeferenced DEM with a CRS and affine transform.",
            context={"dem_path": str(dem_info.path)},
        )
    try:
        dem_crs = CRS.from_wkt(dem_info.crs.wkt)
        dem_wgs84 = cast(
            "tuple[float, float, float, float]",
            tuple(
                float(value)
                for value in transform_bounds(
                    dem_crs,
                    CRS.from_epsg(WGS84_EPSG),
                    *dem_info.bounds,
                    densify_pts=BOUNDS_DENSIFY_POINTS,
                )
            ),
        )
    except (CRSError, RasterioIOError, ValueError) as error:
        raise UnreadableDEMError(
            "Could not transform DEM bounds to WGS84 for the coverage check.",
            hint="Check that the DEM CRS is valid and that PROJ data is usable "
            "(`hypersat version --verbose`).",
            context={"dem_path": str(dem_info.path), "reason": str(error)},
        ) from error

    left, bottom, right, top = scene_bounds_wgs84
    dem_left, dem_bottom, dem_right, dem_top = dem_wgs84
    overlaps = not (right < dem_left or left > dem_right or top < dem_bottom or bottom > dem_top)
    if not overlaps:
        raise UnreadableDEMError(
            "DEM does not overlap the scene footprint estimated from the RPC model.",
            hint="Provide a DEM that covers the scene with a margin (see "
            "docs/data-sources.md). Orthorectification without DEM coverage would "
            "silently degrade to a constant-height warp.",
            context={
                "dem_path": str(dem_info.path),
                "scene_bounds_wgs84": list(scene_bounds_wgs84),
                "dem_bounds_wgs84": list(dem_wgs84),
            },
        )


def _to_rasterio_resampling(method: ResamplingMethod) -> Resampling:
    """Map our enum onto rasterio's ``Resampling``."""
    mapping = {
        ResamplingMethod.NEAREST: Resampling.nearest,
        ResamplingMethod.BILINEAR: Resampling.bilinear,
        ResamplingMethod.CUBIC: Resampling.cubic,
    }
    return mapping[method]


def _authority_label(crs: CRS, epsg: int | None) -> str:
    """Stable CRS token for filenames and reports."""
    if epsg is not None:
        return f"EPSG:{epsg}"
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0]}:{authority[1]}"
    return "crs"


def _output_filename(product_id: str, *, crs_epsg: int | None, resolution: float) -> str:
    """Build ``<id>_ortho_epsg<code>_<res>.tif``."""
    epsg_token = f"epsg{crs_epsg}" if crs_epsg is not None else "crs"
    return f"{product_id}_ortho_{epsg_token}_{format_resolution_token(resolution)}.tif"


def _resolve_bands(info: RasterInfo, bands: Sequence[int] | None) -> tuple[int, ...]:
    """Validate and return 1-based band indices to warp."""
    if bands is None:
        return tuple(range(1, info.band_count + 1))
    out_of_range = sorted(index for index in bands if index < 1 or index > info.band_count)
    if out_of_range:
        raise OrthorectificationError(
            "Requested bands are outside the raster's band range.",
            hint=f"This raster has {info.band_count} band(s); indices are 1-based.",
            context={"out_of_range": out_of_range, "band_count": info.band_count},
        )
    return tuple(bands)


def _require_usable_rpc(info: RasterInfo) -> None:
    """Raise if the product has no usable RPC sensor model."""
    if not info.rpc.available:
        raise MissingRPCMetadataError(
            "Orthorectification requires an RPC sensor model, but the raster has none.",
            hint="Use a product in sensor geometry that ships RPC metadata (EnMAP L1B). "
            "Reprojection is not a substitute. See docs/orthorectification.md.",
            context={"path": str(info.path)},
        )
    if not info.rpc.is_usable:
        raise MissingRPCMetadataError(
            "RPC metadata is present but not usable: " + "; ".join(info.rpc.issues) + ".",
            hint="A usable RPC00B model needs non-degenerate coefficients and non-zero "
            "normalisation scales. Re-download the product before orthorectifying.",
            context={"path": str(info.path), "issues": list(info.rpc.issues)},
        )


def _gdal_transformer_options(
    dem_path: Path,
    options: RpcTransformerOptions,
) -> dict[str, str]:
    """Build the GDAL RPC transformer option dict (string values only)."""
    gdal_options: dict[str, str] = {"RPC_DEM": str(dem_path)}
    if options.rpc_height_scale is not None:
        gdal_options["RPC_HEIGHT_SCALE"] = f"{options.rpc_height_scale:g}"
    if options.rpc_dem_missing_value is not None:
        gdal_options["RPC_DEM_MISSING_VALUE"] = f"{options.rpc_dem_missing_value:g}"
    gdal_options["RPC_DEM_APPLY_VDATUM_SHIFT"] = (
        "TRUE" if options.rpc_dem_apply_vdatum_shift else "FALSE"
    )
    return gdal_options


def _calculate_ortho_grid(
    *,
    rpcs: RPC,
    source_width: int,
    source_height: int,
    destination_crs: CRS,
    resolution: float,
    snap_to_grid: bool,
) -> OutputGrid:
    """Compute the destination transform/shape for an RPC warp."""
    try:
        transform, width, height = calculate_default_transform(
            None,
            destination_crs,
            source_width,
            source_height,
            rpcs=rpcs,
            resolution=resolution,
        )
    except (CRSError, RasterioIOError, ValueError) as error:
        raise OrthorectificationError(
            "Could not calculate the orthorectified output grid from the RPC model.",
            hint="Check that the RPC coefficients are usable and that the target CRS "
            "and resolution are sensible.",
            context={"reason": str(error)},
        ) from error

    effective_resolution = resolution
    if snap_to_grid:
        transform, width, height = aligned_target(transform, width, height, effective_resolution)
        effective_resolution = abs(float(transform.a))

    if width <= 0 or height <= 0:
        raise OrthorectificationError(
            "Calculated orthorectified grid has non-positive dimensions.",
            hint="Check the RPC footprint and requested resolution.",
            context={"width": width, "height": height},
        )
    return OutputGrid(
        crs_wkt=destination_crs.to_wkt(),
        crs_epsg=destination_crs.to_epsg(),
        transform=transform,
        width=int(width),
        height=int(height),
        resolution=float(effective_resolution),
    )


def orthorectify_raster(request: OrthorectifyRequest) -> OrthorectifyResult:
    """Orthorectify a sensor-geometry raster with RPC coefficients and a DEM.

    Args:
        request: Validated orthorectification configuration.

    Returns:
        Paths and configuration summary for the written product.

    Raises:
        MissingRPCMetadataError: If the source has no usable RPC model.
        MissingDEMError: If the DEM path does not exist.
        UnreadableDEMError: If the DEM is unusable or does not cover the scene.
        OrthorectificationError: If grid calculation or warping fails.
        MemoryBudgetExceededError: If the destination cube would exceed the budget.
        RasterWriteError: If the output cannot be written.
        OutputPathError: If the output directory is unusable.
    """
    ensure_usable_proj_data(allow_repair=request.proj_autofix)
    validate_output_directory(request.output.directory, overwrite=request.output.overwrite)

    raster_path, _layout = resolve_raster_path(request.product_path)
    info = inspect_raster(raster_path)
    _require_usable_rpc(info)

    dem_path = request.dem_path
    dem_info = validate_dem(dem_path)
    scene_bounds = rpc_geographic_bounds(info.rpc)
    validate_dem_covers_scene(dem_info, scene_bounds)

    resolved_id = request.product_id or derive_product_id(request.product_path)
    band_indices = _resolve_bands(info, request.bands)
    method = resampling_for_semantics(request.data_semantics, request.resampling)
    if (
        request.data_semantics is DataSemantics.CATEGORICAL
        and request.resampling is not ResamplingMethod.NEAREST
    ):
        logger.info(
            "forcing nearest-neighbour resampling for categorical data",
            extra={"requested": request.resampling.value},
        )

    destination_crs = resolve_target_crs(
        request.target_crs,
        source_crs=CRS.from_epsg(WGS84_EPSG),
        source_bounds=scene_bounds,
    )
    transformer_options = _gdal_transformer_options(dem_path, request.rpc_options)

    try:
        with rasterio.open(raster_path) as dataset:
            if dataset.rpcs is None:
                raise MissingRPCMetadataError(
                    "Orthorectification requires an RPC sensor model, but GDAL did not "
                    "expose one on the open dataset.",
                    hint="Inspect the product with `hypersat inspect --require-rpc`. "
                    "GDAL discards incomplete RPC domains entirely.",
                    context={"path": str(raster_path)},
                )
            rpcs = dataset.rpcs
            grid = _calculate_ortho_grid(
                rpcs=rpcs,
                source_width=dataset.width,
                source_height=dataset.height,
                destination_crs=destination_crs,
                resolution=request.resolution,
                snap_to_grid=request.snap_to_grid,
            )

            sample_dtype = np.dtype(dataset.dtypes[band_indices[0] - 1])
            estimated_bytes = (
                int(grid.width) * int(grid.height) * len(band_indices) * sample_dtype.itemsize
            )
            if estimated_bytes > DEFAULT_READ_BUDGET_BYTES:
                raise MemoryBudgetExceededError(
                    "The orthorectified raster would exceed the in-memory processing budget.",
                    hint="Select fewer bands with --bands, choose a coarser --resolution, "
                    "or crop the scene before orthorectifying.",
                    context={
                        "estimated_bytes": estimated_bytes,
                        "budget_bytes": DEFAULT_READ_BUDGET_BYTES,
                        "width": grid.width,
                        "height": grid.height,
                        "bands": len(band_indices),
                    },
                )

            destination = np.zeros(
                (len(band_indices), grid.height, grid.width),
                dtype=sample_dtype,
            )
            src_nodata = info.nodata
            dst_nodata = request.nodata if request.nodata is not None else src_nodata
            rio_resampling = _to_rasterio_resampling(method)

            for out_index, band_index in enumerate(band_indices):
                band_dtype = np.dtype(dataset.dtypes[band_index - 1])
                if band_dtype != sample_dtype:
                    raise OrthorectificationError(
                        "Mixed band dtypes are not supported in one orthorectify call.",
                        hint="Orthorectify homogeneous bands together, or run once per dtype.",
                        context={
                            "first_dtype": str(sample_dtype),
                            "band": band_index,
                            "dtype": str(band_dtype),
                        },
                    )
                try:
                    # Pass a dataset band, not a NumPy array: rasterio requires src_crs for
                    # array sources even when an RPC model supplies the geometry.
                    reproject(
                        source=rasterio.band(dataset, band_index),
                        destination=destination[out_index],
                        rpcs=rpcs,
                        dst_transform=grid.transform,
                        dst_crs=destination_crs,
                        src_nodata=src_nodata,
                        dst_nodata=dst_nodata,
                        resampling=rio_resampling,
                        warp_mem_limit=request.warp_memory_mb,
                        **transformer_options,
                    )
                except (RasterioIOError, CRSError, ValueError, WarpOperationError) as error:
                    raise OrthorectificationError(
                        "Orthorectification failed while warping pixels.",
                        hint="Check the RPC model, DEM coverage and that PROJ data is usable. "
                        "See docs/orthorectification.md.",
                        context={
                            "reason": str(error),
                            "path": str(raster_path),
                            "dem_path": str(dem_path),
                            "band": band_index,
                        },
                    ) from error
    except MemoryBudgetExceededError:
        raise
    except MissingRPCMetadataError:
        raise
    except OrthorectificationError:
        raise
    except (RasterioIOError, CRSError, ValueError, WarpOperationError) as error:
        raise OrthorectificationError(
            "Orthorectification failed while opening or preparing the source.",
            hint="Check that the product opens with `hypersat inspect` and that PROJ "
            "data is usable.",
            context={"reason": str(error), "path": str(raster_path)},
        ) from error

    authority = _authority_label(destination_crs, grid.crs_epsg)
    output_path = request.output.directory / _output_filename(
        resolved_id,
        crs_epsg=grid.crs_epsg,
        resolution=grid.resolution,
    )
    descriptions = tuple(info.bands[index - 1].description or "" for index in band_indices)
    wavelengths = tuple(info.bands[index - 1].wavelength_nm for index in band_indices)
    metadata = RasterMetadata(
        crs_wkt=grid.crs_wkt,
        transform=tuple(float(value) for value in tuple(grid.transform)[:6]),
        nodata=dst_nodata,
        band_descriptions=descriptions,
        wavelengths_nm=wavelengths,
        dataset_tags={
            "ORTHO_TARGET_CRS": authority,
            "ORTHO_RESOLUTION": f"{grid.resolution:g}",
            "ORTHO_RESAMPLING": method.value,
            "ORTHO_DATA_SEMANTICS": request.data_semantics.value,
            "ORTHO_DEM": str(dem_path),
            "ORTHO_ERROR_THRESHOLD_PX": f"{request.error_threshold_px:g}",
            "ORTHO_WARP_MEMORY_MB": str(request.warp_memory_mb),
            "ORTHO_SNAPPED": "true" if request.snap_to_grid else "false",
            **{f"ORTHO_{key}": value for key, value in transformer_options.items()},
        },
    )
    write_array(
        output_path,
        cast("npt.NDArray[Any]", destination),
        metadata=metadata,
        overwrite=request.output.overwrite,
    )

    logger.info(
        "wrote orthorectified raster",
        extra={
            "path": str(output_path),
            "crs": authority,
            "resolution": grid.resolution,
            "width": grid.width,
            "height": grid.height,
            "resampling": method.value,
            "dem_path": str(dem_path),
        },
    )
    return OrthorectifyResult(
        path=output_path,
        crs_authority=authority,
        resolution=grid.resolution,
        width=grid.width,
        height=grid.height,
        band_indices=band_indices,
        resampling=method,
        dem_path=dem_path,
        transformer_options=transformer_options,
        product_id=resolved_id,
    )
