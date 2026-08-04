"""CRS reprojection and optional reference-grid alignment.

This is map-to-map warping only. Imagery that is still in sensor geometry (RPC, no
affine CRS) must go through orthorectification instead; this module refuses that case
rather than silently substituting a plain reprojection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError, RasterioIOError
from rasterio.transform import Affine
from rasterio.warp import (
    aligned_target,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

from hypersat.exceptions import (
    MemoryBudgetExceededError,
    MissingGeoreferencingError,
    ReprojectionError,
)
from hypersat.io.environment import ensure_usable_proj_data
from hypersat.io.files import derive_product_id
from hypersat.io.inspect import inspect_raster, resolve_raster_path
from hypersat.io.reader import DEFAULT_READ_BUDGET_BYTES
from hypersat.io.writer import write_array
from hypersat.logging_config import get_logger
from hypersat.models.config import (
    DataSemantics,
    ReprojectRequest,
    ResamplingMethod,
)
from hypersat.models.product import RasterInfo
from hypersat.models.raster import RasterMetadata
from hypersat.processing.validation import validate_output_directory

__all__ = [
    "OutputGrid",
    "ReprojectResult",
    "calculate_output_grid",
    "format_resolution_token",
    "reproject_raster",
    "resampling_for_semantics",
    "resolve_target_crs",
    "select_utm_epsg",
    "wgs84_bounds",
]

logger = get_logger(__name__)

WGS84_EPSG = 4326
UTM_LATITUDE_MAX_NORTH = 84.0
UTM_LATITUDE_MIN_SOUTH = -80.0
LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0
# Refuse auto-UTM when corner zones differ by more than this many zone numbers.
MAX_UTM_ZONE_SPAN = 1
BOUNDS_DENSIFY_POINTS = 21
_AUTO_CRS = "auto"


@dataclass(frozen=True, slots=True)
class OutputGrid:
    """Destination georeferencing for a reprojected raster."""

    crs_wkt: str
    crs_epsg: int | None
    transform: Affine
    width: int
    height: int
    resolution: float


@dataclass(frozen=True, slots=True)
class ReprojectResult:
    """Outcome of a successful reprojection."""

    path: Path
    crs_authority: str
    resolution: float
    width: int
    height: int
    band_indices: tuple[int, ...]
    resampling: ResamplingMethod
    snapped: bool
    reference_raster: Path | None
    product_id: str


def select_utm_epsg(longitude: float, latitude: float) -> int:
    """Return the EPSG code of the UTM zone containing a WGS84 point.

    Args:
        longitude: Longitude in degrees, ``[-180, 180]``.
        latitude: Latitude in degrees.

    Returns:
        EPSG code (326xx northern hemisphere, 327xx southern).

    Raises:
        ValueError: If the point is outside the UTM latitude domain or longitude range.
    """
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ValueError("longitude and latitude must be finite")
    if longitude < LONGITUDE_MIN or longitude > LONGITUDE_MAX:
        raise ValueError(f"longitude must be in [-180, 180], got {longitude}")
    if latitude > UTM_LATITUDE_MAX_NORTH or latitude < UTM_LATITUDE_MIN_SOUTH:
        raise ValueError(
            f"latitude {latitude} is outside the UTM domain "
            f"[{UTM_LATITUDE_MIN_SOUTH}, {UTM_LATITUDE_MAX_NORTH}]; "
            "use an explicit polar CRS instead of auto"
        )
    zone = int((longitude + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if latitude >= 0.0 else 32700
    return base + zone


def format_resolution_token(resolution: float) -> str:
    """Format a ground sample distance for output filenames (``30m``, ``0p5m``)."""
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(f"resolution must be a positive finite number, got {resolution}")
    if float(resolution).is_integer():
        return f"{int(resolution)}m"
    text = f"{resolution:.6f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}m"


def resampling_for_semantics(
    semantics: DataSemantics,
    requested: ResamplingMethod,
) -> ResamplingMethod:
    """Resolve the resampling method for the data kind.

    Categorical rasters always use nearest-neighbour: averaging class codes invents
    meaningless intermediate values.
    """
    if semantics is DataSemantics.CATEGORICAL:
        return ResamplingMethod.NEAREST
    return requested


def _crs_from_spec(spec: str) -> CRS:
    """Parse an authority code or WKT-like CRS string."""
    cleaned = spec.strip()
    try:
        if cleaned.upper().startswith("EPSG:"):
            return CRS.from_epsg(int(cleaned.split(":", 1)[1]))
        if cleaned.isdigit():
            return CRS.from_epsg(int(cleaned))
        return CRS.from_user_input(cleaned)
    except (CRSError, ValueError) as error:
        raise ReprojectionError(
            f"Could not interpret target CRS {spec!r}.",
            hint="Pass an authority code such as EPSG:32633, or 'auto' for UTM.",
            context={"target_crs": spec, "reason": str(error)},
        ) from error


def wgs84_bounds(
    crs: CRS,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Transform ``(left, bottom, right, top)`` into WGS84 geographic coordinates."""
    try:
        wgs84 = CRS.from_epsg(WGS84_EPSG)
        transformed = transform_bounds(crs, wgs84, *bounds, densify_pts=BOUNDS_DENSIFY_POINTS)
        return cast(
            "tuple[float, float, float, float]",
            tuple(float(value) for value in transformed),
        )
    except (CRSError, RasterioIOError, ValueError) as error:
        raise ReprojectionError(
            "Could not transform the source bounds to WGS84.",
            hint="Check that the source CRS is valid and that PROJ data is usable "
            "(`hypersat version --verbose`).",
            context={"reason": str(error)},
        ) from error


def resolve_target_crs(
    spec: str,
    *,
    source_crs: CRS,
    source_bounds: tuple[float, float, float, float],
) -> CRS:
    """Resolve ``auto`` or an explicit CRS string to a ``CRS`` object.

    ``auto`` selects the UTM zone of the scene centre and refuses scenes that span too
    many UTM zones or lie outside the UTM latitude domain.
    """
    cleaned = spec.strip()
    if cleaned.lower() != _AUTO_CRS:
        return _crs_from_spec(cleaned)

    left, bottom, right, top = wgs84_bounds(source_crs, source_bounds)
    centre_lon = (left + right) / 2.0
    centre_lat = (bottom + top) / 2.0
    try:
        epsg = select_utm_epsg(centre_lon, centre_lat)
    except ValueError as error:
        raise ReprojectionError(
            "Automatic UTM selection is not appropriate for this scene.",
            hint="Pass an explicit --target-crs (for polar scenes, a polar stereographic "
            "CRS). 'auto' only covers the UTM latitude domain.",
            context={
                "centre_lon": centre_lon,
                "centre_lat": centre_lat,
                "reason": str(error),
            },
        ) from error

    corners = (
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),
        (centre_lon, centre_lat),
    )
    zones: set[int] = set()
    for lon, lat in corners:
        try:
            zones.add(select_utm_epsg(lon, lat) % 100)
        except ValueError as error:
            raise ReprojectionError(
                "Automatic UTM selection is not appropriate for this scene.",
                hint="Pass an explicit --target-crs.",
                context={"reason": str(error), "longitude": lon, "latitude": lat},
            ) from error
    if max(zones) - min(zones) > MAX_UTM_ZONE_SPAN:
        raise ReprojectionError(
            "Scene spans too many UTM zones for automatic CRS selection.",
            hint="Pass an explicit --target-crs covering the area of interest.",
            context={"zones": sorted(zones), "max_span": MAX_UTM_ZONE_SPAN},
        )
    return CRS.from_epsg(epsg)


def _resolution_from_transform(transform: Affine) -> float:
    """Return a single positive pixel size; prefer X when X and Y differ slightly."""
    x_res = abs(float(transform.a))
    y_res = abs(float(transform.e))
    if not math.isfinite(x_res) or x_res <= 0.0:
        raise ReprojectionError(
            "Could not determine a positive output resolution.",
            hint="Pass --resolution explicitly.",
            context={"transform": list(transform)[:6]},
        )
    if math.isfinite(y_res) and y_res > 0.0 and not math.isclose(x_res, y_res, rel_tol=1e-6):
        logger.warning(
            "output X and Y resolutions differ; using X for naming and snapping",
            extra={"x_resolution": x_res, "y_resolution": y_res},
        )
    return x_res


def _grid_from_reference(
    reference_path: Path,
    *,
    source_crs: CRS,
    source_bounds: tuple[float, float, float, float],
    target_crs: CRS | None,
) -> OutputGrid:
    """Build an output grid snapped to a reference raster's origin and resolution."""
    try:
        with rasterio.open(reference_path) as reference:
            if reference.crs is None or reference.transform.is_identity:
                raise ReprojectionError(
                    "The reference raster is not georeferenced.",
                    hint="Choose a map-geometry GeoTIFF with a CRS and affine transform.",
                    context={"reference_raster": str(reference_path)},
                )
            ref_crs = reference.crs
            if target_crs is not None and not target_crs.equals(ref_crs):
                raise ReprojectionError(
                    "target_crs does not match the reference raster CRS.",
                    hint="Omit --target-crs when aligning to a reference, or pass the "
                    "same CRS the reference uses.",
                    context={
                        "target_crs": target_crs.to_string(),
                        "reference_crs": ref_crs.to_string(),
                    },
                )
            resolution = _resolution_from_transform(reference.transform)
            try:
                left, bottom, right, top = transform_bounds(
                    source_crs,
                    ref_crs,
                    *source_bounds,
                    densify_pts=BOUNDS_DENSIFY_POINTS,
                )
            except (CRSError, RasterioIOError, ValueError) as error:
                raise ReprojectionError(
                    "Could not transform source bounds into the reference CRS.",
                    hint="Check that source and reference CRSs are compatible.",
                    context={"reason": str(error)},
                ) from error

            if right <= left or top <= bottom:
                raise ReprojectionError(
                    "Transformed source bounds are empty or inverted.",
                    hint="Verify the source georeferencing.",
                    context={"bounds": [left, bottom, right, top]},
                )

            ref_left = float(reference.bounds.left)
            ref_bottom = float(reference.bounds.bottom)
            ref_right = float(reference.bounds.right)
            ref_top = float(reference.bounds.top)
            if right < ref_left or left > ref_right or top < ref_bottom or bottom > ref_top:
                raise ReprojectionError(
                    "Source footprint does not overlap the reference raster bounds.",
                    hint="Choose a reference that covers the scene, or omit "
                    "--reference-raster and set --target-crs / --resolution instead.",
                    context={
                        "source_bounds": [left, bottom, right, top],
                        "reference_bounds": [ref_left, ref_bottom, ref_right, ref_top],
                    },
                )

            # Snap the output window to the reference grid origin.
            origin_x = float(reference.transform.c)
            origin_y = float(reference.transform.f)
            col_off = math.floor((left - origin_x) / resolution)
            row_off = math.floor((origin_y - top) / resolution)
            out_left = origin_x + col_off * resolution
            out_top = origin_y - row_off * resolution
            width = max(1, math.ceil((right - out_left) / resolution))
            height = max(1, math.ceil((out_top - bottom) / resolution))
            transform = Affine(resolution, 0.0, out_left, 0.0, -resolution, out_top)
            return OutputGrid(
                crs_wkt=ref_crs.to_wkt(),
                crs_epsg=ref_crs.to_epsg(),
                transform=transform,
                width=width,
                height=height,
                resolution=resolution,
            )
    except RasterioIOError as error:
        raise ReprojectionError(
            "The reference raster could not be opened.",
            hint="Check the path and that the file is a readable GeoTIFF.",
            context={"reference_raster": str(reference_path), "reason": str(error)},
        ) from error


def calculate_output_grid(
    *,
    source_crs: CRS,
    source_bounds: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    destination_crs: CRS,
    resolution: float | None,
    snap_to_grid: bool,
) -> OutputGrid:
    """Compute the destination transform and shape for a map-to-map warp."""
    try:
        transform, width, height = calculate_default_transform(
            source_crs,
            destination_crs,
            source_width,
            source_height,
            *source_bounds,
            resolution=resolution,
        )
    except (CRSError, RasterioIOError, ValueError) as error:
        raise ReprojectionError(
            "Could not calculate the output grid.",
            hint="Check the source georeferencing, target CRS and resolution.",
            context={"reason": str(error)},
        ) from error

    effective_resolution = (
        resolution if resolution is not None else _resolution_from_transform(transform)
    )
    if snap_to_grid:
        transform, width, height = aligned_target(transform, width, height, effective_resolution)
        effective_resolution = _resolution_from_transform(transform)

    if width <= 0 or height <= 0:
        raise ReprojectionError(
            "Calculated output grid has non-positive dimensions.",
            hint="Check the source bounds and requested resolution.",
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


def _output_filename(
    product_id: str,
    *,
    aligned: bool,
    crs_epsg: int | None,
    resolution: float,
) -> str:
    """Build ``<id>_(reprojected|aligned)_epsg<code>_<res>.tif``."""
    content = "aligned" if aligned else "reprojected"
    epsg_token = f"epsg{crs_epsg}" if crs_epsg is not None else "crs"
    return f"{product_id}_{content}_{epsg_token}_{format_resolution_token(resolution)}.tif"


def _resolve_bands(info: RasterInfo, bands: Sequence[int] | None) -> tuple[int, ...]:
    """Validate and return 1-based band indices to warp."""
    if bands is None:
        return tuple(range(1, info.band_count + 1))
    out_of_range = sorted(index for index in bands if index < 1 or index > info.band_count)
    if out_of_range:
        raise ReprojectionError(
            "Requested bands are outside the raster's band range.",
            hint=f"This raster has {info.band_count} band(s); indices are 1-based.",
            context={"out_of_range": out_of_range, "band_count": info.band_count},
        )
    return tuple(bands)


def reproject_raster(request: ReprojectRequest) -> ReprojectResult:
    """Reproject a map-geometry raster, optionally snapping to a reference grid.

    Args:
        request: Validated reprojection configuration.

    Returns:
        Paths and grid summary for the written product.

    Raises:
        MissingGeoreferencingError: If the source lacks affine map georeferencing.
        ReprojectionError: If CRS resolution, grid calculation or warping fails.
        MemoryBudgetExceededError: If the destination cube would exceed the read budget.
        RasterWriteError: If the output cannot be written.
        OutputPathError: If the output directory is unusable.
    """
    ensure_usable_proj_data(allow_repair=request.proj_autofix)
    validate_output_directory(request.output.directory, overwrite=request.output.overwrite)

    raster_path, _layout = resolve_raster_path(request.product_path)
    info = inspect_raster(raster_path)
    if not info.has_affine_georeferencing or info.bounds is None:
        raise MissingGeoreferencingError(
            "Reprojection requires a CRS and a non-identity affine transform.",
            hint="Sensor-geometry products with RPC metadata need orthorectification "
            "(milestone 8), not reprojection. Georeference the raster first, or pass a "
            "map-geometry GeoTIFF.",
            context={
                "path": str(raster_path),
                "crs_defined": info.crs.is_defined,
                "has_rpc": info.rpc.available,
            },
        )

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

    try:
        source_crs = (
            CRS.from_wkt(info.crs.wkt)
            if info.crs.wkt
            else CRS.from_user_input(info.crs.authority_code or "")
        )
    except (CRSError, ValueError) as error:
        raise ReprojectionError(
            "The source CRS could not be interpreted.",
            hint="Inspect the product with `hypersat inspect` and check PROJ data.",
            context={"reason": str(error)},
        ) from error

    source_bounds = info.bounds
    reference_path = (
        None if request.reference_raster is None else request.reference_raster.expanduser()
    )

    if reference_path is not None:
        # ``auto`` means "inherit CRS from the reference", not "pick a UTM zone".
        override_crs = (
            None
            if request.target_crs.strip().lower() == _AUTO_CRS
            else _crs_from_spec(request.target_crs)
        )
        grid = _grid_from_reference(
            reference_path,
            source_crs=source_crs,
            source_bounds=source_bounds,
            target_crs=override_crs,
        )
        destination_crs = CRS.from_wkt(grid.crs_wkt)
        snapped = True
    else:
        destination_crs = resolve_target_crs(
            request.target_crs,
            source_crs=source_crs,
            source_bounds=source_bounds,
        )
        grid = calculate_output_grid(
            source_crs=source_crs,
            source_bounds=source_bounds,
            source_width=info.width,
            source_height=info.height,
            destination_crs=destination_crs,
            resolution=request.resolution,
            snap_to_grid=request.snap_to_grid,
        )
        snapped = request.snap_to_grid

    sample_dtype = np.dtype(info.dtypes[band_indices[0] - 1])
    estimated_bytes = int(grid.width) * int(grid.height) * len(band_indices) * sample_dtype.itemsize
    if estimated_bytes > DEFAULT_READ_BUDGET_BYTES:
        raise MemoryBudgetExceededError(
            "The reprojected raster would exceed the in-memory processing budget.",
            hint="Select fewer bands with --bands, choose a coarser --resolution, or "
            "crop the scene before reprojecting.",
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

    try:
        with rasterio.open(raster_path) as dataset:
            source_transform = dataset.transform
            for out_index, band_index in enumerate(band_indices):
                band_dtype = np.dtype(dataset.dtypes[band_index - 1])
                if band_dtype != sample_dtype:
                    raise ReprojectionError(
                        "Mixed band dtypes are not supported in one reproject call.",
                        hint="Reproject homogeneous bands together, or run once per dtype.",
                        context={
                            "first_dtype": str(sample_dtype),
                            "band": band_index,
                            "dtype": str(band_dtype),
                        },
                    )
                source = np.array(dataset.read(band_index), copy=True)
                reproject(
                    source=source,
                    destination=destination[out_index],
                    src_transform=source_transform,
                    src_crs=source_crs,
                    dst_transform=grid.transform,
                    dst_crs=destination_crs,
                    src_nodata=src_nodata,
                    dst_nodata=dst_nodata,
                    resampling=rio_resampling,
                )
    except MemoryBudgetExceededError:
        raise
    except (RasterioIOError, CRSError, ValueError) as error:
        raise ReprojectionError(
            "Reprojection failed while warping pixels.",
            hint="Check source georeferencing, target CRS and that PROJ data is usable.",
            context={"reason": str(error), "path": str(raster_path)},
        ) from error

    authority = _authority_label(destination_crs, grid.crs_epsg)
    filename = _output_filename(
        resolved_id,
        aligned=reference_path is not None,
        crs_epsg=grid.crs_epsg,
        resolution=grid.resolution,
    )
    output_path = request.output.directory / filename

    descriptions = tuple(info.bands[index - 1].description or "" for index in band_indices)
    wavelengths = tuple(info.bands[index - 1].wavelength_nm for index in band_indices)
    metadata = RasterMetadata(
        crs_wkt=grid.crs_wkt,
        transform=tuple(float(value) for value in tuple(grid.transform)[:6]),
        nodata=dst_nodata,
        band_descriptions=descriptions,
        wavelengths_nm=wavelengths,
        dataset_tags={
            "REPROJECT_TARGET_CRS": authority,
            "REPROJECT_RESOLUTION": f"{grid.resolution:g}",
            "REPROJECT_RESAMPLING": method.value,
            "REPROJECT_DATA_SEMANTICS": request.data_semantics.value,
            "REPROJECT_SNAPPED": "true" if snapped else "false",
            **({"REPROJECT_REFERENCE": str(reference_path)} if reference_path is not None else {}),
        },
    )
    write_array(
        output_path,
        cast("npt.NDArray[Any]", destination),
        metadata=metadata,
        overwrite=request.output.overwrite,
    )

    logger.info(
        "wrote reprojected raster",
        extra={
            "path": str(output_path),
            "crs": authority,
            "resolution": grid.resolution,
            "width": grid.width,
            "height": grid.height,
            "resampling": method.value,
        },
    )
    return ReprojectResult(
        path=output_path,
        crs_authority=authority,
        resolution=grid.resolution,
        width=grid.width,
        height=grid.height,
        band_indices=band_indices,
        resampling=method,
        snapped=snapped,
        reference_raster=reference_path,
        product_id=resolved_id,
    )
