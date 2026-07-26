"""Read-only inspection of rasters and satellite-product directories.

This is the only place that opens a dataset for the inspection command, and it reads
**metadata only** - never pixels. A full EnMAP cube is several gigabytes, so inspection
must stay fast and memory-flat, which also means no statistics are reported here.

All rasterio access is wrapped so that library-specific exceptions become
:class:`hypersat.exceptions.HyperSatError` subclasses carrying an actionable hint.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import CRSError, NotGeoreferencedWarning, RasterioIOError

from hypersat.exceptions import ProductStructureError, RasterReadError
from hypersat.io.environment import describe_environment, ensure_usable_proj_data
from hypersat.io.files import describe_file, directory_size_bytes, format_bytes
from hypersat.logging_config import get_logger
from hypersat.models.environment import ProjDataStatus
from hypersat.models.product import (
    BandInfo,
    CRSInfo,
    FileInfo,
    InputKind,
    InspectionResult,
    ProductLayout,
    RasterInfo,
    RPCInfo,
    WavelengthSource,
)

__all__ = [
    "METADATA_EXTENSIONS",
    "RASTER_EXTENSIONS",
    "extract_wavelength_nm",
    "inspect_input",
    "inspect_raster",
    "resolve_raster_path",
    "scan_product_directory",
]

logger = get_logger(__name__)

RASTER_EXTENSIONS = frozenset(
    {".tif", ".tiff", ".jp2", ".img", ".bsq", ".bil", ".bip", ".dat", ".vrt", ".he5", ".hdf"}
)
"""Extensions treated as candidate imagery inside a product directory.

Deliberately conservative: an unrecognised extension leads to an actionable "pass the
raster directly" error, which is better than opening an arbitrary file and guessing.
"""

METADATA_EXTENSIONS = frozenset({".xml", ".json", ".hdr", ".met", ".txt", ".gfs", ".rpc"})

PREFERRED_RASTER_TOKENS = ("spectral_image", "spectral-image")
"""EnMAP names its imagery ``*-SPECTRAL_IMAGE.*``; preferred when several candidates exist."""

_RPC_COEFFICIENT_FIELDS = ("line_num_coeff", "line_den_coeff", "samp_num_coeff", "samp_den_coeff")
_RPC_REQUIRED_SCALARS = (
    "line_off",
    "samp_off",
    "lat_off",
    "long_off",
    "height_off",
    "line_scale",
    "samp_scale",
    "lat_scale",
    "long_scale",
    "height_scale",
)
_RPC_SCALE_FIELDS = ("line_scale", "samp_scale", "lat_scale", "long_scale", "height_scale")
RPC_COEFFICIENTS_PER_SET = 20
"""An RPC00B polynomial has 20 cubic terms."""

_WAVELENGTH_TAG_KEYS = (
    "wavelength",
    "wavelengths",
    "central_wavelength",
    "centre_wavelength",
    "center_wavelength",
)
_WAVELENGTH_UNIT_KEYS = ("wavelength_units", "wavelength_unit")
_NANOMETRES_PER_UNIT = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "nanometre": 1.0,
    "nanometres": 1.0,
    "um": 1000.0,
    "\u00b5m": 1000.0,
    "\u03bcm": 1000.0,
    "micron": 1000.0,
    "microns": 1000.0,
    "micrometer": 1000.0,
    "micrometers": 1000.0,
    "micrometre": 1000.0,
    "micrometres": 1000.0,
}
_DESCRIPTION_WAVELENGTH_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nm|nanometre?s?|nanometer?s?|um|\u00b5m|\u03bcm|"
    r"micron?s?|micrometre?s?|micrometer?s?)\b",
    re.IGNORECASE,
)
_MICROMETRE_HEURISTIC_MAX = 100.0
"""Below this, a unit-less wavelength is interpreted as micrometres rather than nanometres.

Optical/SWIR instruments report roughly 0.4-2.5 um or 400-2500 nm, so the two ranges do
not overlap and the heuristic is unambiguous for this domain.
"""
_PLAUSIBLE_NM_RANGE = (100.0, 30000.0)
"""Values outside this range are rejected rather than reported as a wavelength."""


def _to_float(value: object) -> float | None:
    """Convert a metadata value to ``float``, returning ``None`` when it is not numeric."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def extract_wavelength_nm(
    band_tags: Mapping[str, str],
    imagery_tags: Mapping[str, str] | None = None,
    description: str | None = None,
) -> tuple[float | None, WavelengthSource | None]:
    """Determine a band's centre wavelength in nanometres.

    Sources are tried in decreasing order of trustworthiness: an explicit band tag, then
    GDAL's ``IMAGERY`` domain, then the free-text band description. The source is returned
    alongside the value so a report can show how the number was obtained.

    Args:
        band_tags: Tags of the band's default metadata domain.
        imagery_tags: Tags of the band's ``IMAGERY`` domain, if read.
        description: Band description text, if any.

    Returns:
        ``(wavelength_nm, source)``, or ``(None, None)`` when nothing usable was found.
    """
    lowered = {key.lower(): value for key, value in band_tags.items()}

    unit_factor: float | None = None
    for unit_key in _WAVELENGTH_UNIT_KEYS:
        raw_unit = lowered.get(unit_key)
        if raw_unit:
            unit_factor = _NANOMETRES_PER_UNIT.get(raw_unit.strip().lower())
            break

    for key in _WAVELENGTH_TAG_KEYS:
        value = _to_float(lowered.get(key))
        if value is None or value <= 0.0:
            continue
        factor = unit_factor
        if factor is None:
            factor = 1000.0 if value < _MICROMETRE_HEURISTIC_MAX else 1.0
        nanometres = value * factor
        if _PLAUSIBLE_NM_RANGE[0] <= nanometres <= _PLAUSIBLE_NM_RANGE[1]:
            return nanometres, WavelengthSource.BAND_TAG

    if imagery_tags:
        imagery_lowered = {key.lower(): value for key, value in imagery_tags.items()}
        micrometres = _to_float(imagery_lowered.get("central_wavelength_um"))
        if micrometres is not None and micrometres > 0.0:
            nanometres = micrometres * 1000.0
            if _PLAUSIBLE_NM_RANGE[0] <= nanometres <= _PLAUSIBLE_NM_RANGE[1]:
                return nanometres, WavelengthSource.IMAGERY_DOMAIN

    if description:
        match = _DESCRIPTION_WAVELENGTH_PATTERN.search(description)
        if match:
            value = float(match.group("value"))
            factor = _NANOMETRES_PER_UNIT.get(match.group("unit").lower(), 1.0)
            nanometres = value * factor
            if _PLAUSIBLE_NM_RANGE[0] <= nanometres <= _PLAUSIBLE_NM_RANGE[1]:
                return nanometres, WavelengthSource.DESCRIPTION

    return None, None


def scan_product_directory(directory: Path) -> ProductLayout:
    """List the raster and metadata files inside a product directory.

    No mission product specification is interpreted here; this only reports what is on
    disk. Parsing EnMAP's metadata XML is a later milestone.

    Args:
        directory: Product root directory.

    Returns:
        A populated :class:`~hypersat.models.product.ProductLayout`.
    """
    raster_candidates: list[Path] = []
    metadata_files: list[Path] = []
    other_files = 0

    for entry in sorted(directory.rglob("*")):
        if not entry.is_file() or entry.is_symlink():
            continue
        suffix = entry.suffix.lower()
        if suffix in RASTER_EXTENSIONS:
            raster_candidates.append(entry)
        elif suffix in METADATA_EXTENSIONS:
            metadata_files.append(entry)
        else:
            other_files += 1

    total = directory_size_bytes(directory)
    return ProductLayout(
        root=directory,
        raster_candidates=raster_candidates,
        metadata_files=metadata_files,
        other_files_count=other_files,
        total_size_bytes=total,
        total_size_human=format_bytes(total),
    )


def resolve_raster_path(input_path: Path) -> tuple[Path, ProductLayout | None]:
    """Resolve the raster to inspect from a file or product-directory path.

    Args:
        input_path: Raster file or product directory.

    Returns:
        ``(raster_path, layout)``; ``layout`` is ``None`` when the input was a file.

    Raises:
        ProductStructureError: If the path does not exist, or a directory contains no
            unambiguous imagery.
    """
    if not input_path.exists():
        raise ProductStructureError(
            "Input path does not exist.",
            hint="Pass --input with a path to a raster file or a product directory. "
            "See data/README.md for how to obtain an EnMAP sample product.",
            context={"input_path": str(input_path)},
        )

    if input_path.is_file():
        return input_path, None

    layout = scan_product_directory(input_path)
    if not layout.raster_candidates:
        raise ProductStructureError(
            "No raster file was found in the product directory.",
            hint="Expected one of the extensions "
            f"{sorted(RASTER_EXTENSIONS)}. If the imagery uses a different extension, "
            "pass the raster file directly with --input.",
            context={
                "product_dir": str(input_path),
                "metadata_files_found": len(layout.metadata_files),
                "other_files_found": layout.other_files_count,
            },
        )

    preferred = [
        candidate
        for candidate in layout.raster_candidates
        if any(token in candidate.name.lower() for token in PREFERRED_RASTER_TOKENS)
    ]
    shortlist = preferred or layout.raster_candidates
    if len(shortlist) > 1:
        raise ProductStructureError(
            "The product directory contains several candidate rasters; "
            "cannot decide which one is the imagery.",
            hint="Pass the raster file explicitly with --input.",
            context={
                "product_dir": str(input_path),
                "candidates": [str(candidate) for candidate in shortlist],
            },
        )

    resolved = shortlist[0]
    logger.debug(
        "resolved product raster",
        extra={"product_dir": str(input_path), "raster": str(resolved)},
    )
    return resolved, layout


def _crs_info(dataset: rasterio.DatasetReader, warnings_out: list[str]) -> CRSInfo:
    """Build :class:`CRSInfo`, degrading gracefully if the PROJ database is unusable."""
    try:
        crs = dataset.crs
    except CRSError as error:  # pragma: no cover - needs a broken PROJ database
        warnings_out.append(f"CRS could not be read: {error}")
        return CRSInfo(is_defined=False)

    if crs is None:
        return CRSInfo(is_defined=False)

    epsg: int | None = None
    authority: str | None = None
    linear_units: str | None = None
    units_factor: float | None = None
    try:
        epsg = crs.to_epsg()
        authority = f"EPSG:{epsg}" if epsg is not None else None
        if crs.is_projected:
            linear_units = str(crs.linear_units)
            units_factor = float(crs.units_factor[1])
    except CRSError as error:  # pragma: no cover - needs a broken PROJ database
        warnings_out.append(f"CRS is defined but could not be interpreted: {error}")

    return CRSInfo(
        is_defined=True,
        authority_code=authority,
        epsg=epsg,
        wkt=str(crs.to_wkt()),
        is_geographic=bool(crs.is_geographic),
        is_projected=bool(crs.is_projected),
        linear_units=linear_units,
        linear_units_factor=units_factor,
    )


def _rpc_info(dataset: rasterio.DatasetReader) -> RPCInfo:
    """Summarise the RPC sensor model and whether it is usable for warping.

    GDAL normalises the ``RPC`` metadata domain on both read and write: an incomplete set
    is discarded entirely and short coefficient lists are zero-padded to 20 terms. The
    checks below therefore look for values that would make the coordinate transformation
    undefined rather than merely counting terms.
    """
    rpcs = dataset.rpcs
    if not rpcs:
        return RPCInfo(available=False)

    values = dict(rpcs.to_dict())
    counts: dict[str, int] = {}
    issues: list[str] = []

    for field in _RPC_COEFFICIENT_FIELDS:
        raw = values.get(field) or []
        coefficients = [value for value in (_to_float(item) for item in raw) if value is not None]
        counts[field] = len(coefficients)
        if len(coefficients) != RPC_COEFFICIENTS_PER_SET:
            issues.append(
                f"{field} has {len(coefficients)} usable terms, expected {RPC_COEFFICIENTS_PER_SET}"
            )
        elif not any(coefficients):
            issues.append(f"{field} is entirely zero, so the polynomial is degenerate")

    scalars = {name: _to_float(values.get(name)) for name in _RPC_REQUIRED_SCALARS}
    issues.extend(f"{name} is missing" for name, value in sorted(scalars.items()) if value is None)
    issues.extend(
        f"{name} is zero, which leaves the RPC normalisation undefined"
        for name in _RPC_SCALE_FIELDS
        if scalars.get(name) == 0.0
    )

    return RPCInfo(
        available=True,
        is_usable=not issues,
        issues=issues,
        coefficient_counts=counts,
        err_bias=_to_float(values.get("err_bias")),
        err_rand=_to_float(values.get("err_rand")),
        **scalars,
    )


def _band_info(dataset: rasterio.DatasetReader, index: int) -> BandInfo:
    """Collect metadata for one 1-based band index."""
    position = index - 1
    band_tags: Mapping[str, str] = dict(dataset.tags(index))
    imagery_tags: Mapping[str, str] = dict(dataset.tags(index, ns="IMAGERY"))
    description = dataset.descriptions[position]
    wavelength, wavelength_source = extract_wavelength_nm(band_tags, imagery_tags, description)

    nodata = _to_float(dataset.nodatavals[position])
    block_shape = dataset.block_shapes[position]
    return BandInfo(
        index=index,
        dtype=str(dataset.dtypes[position]),
        nodata=None if nodata is not None and np.isnan(nodata) else nodata,
        nodata_is_nan=nodata is not None and bool(np.isnan(nodata)),
        description=description,
        units=dataset.units[position],
        wavelength_nm=wavelength,
        wavelength_source=wavelength_source,
        block_shape=(int(block_shape[0]), int(block_shape[1])),
        scale=_to_float(dataset.scales[position]),
        offset=_to_float(dataset.offsets[position]),
        color_interpretation=str(dataset.colorinterp[position].name),
        mask_flags=[str(flag.name) for flag in dataset.mask_flag_enums[position]],
        metadata_domains=[str(domain) for domain in dataset.tag_namespaces(index)],
        tags={str(key): str(value) for key, value in band_tags.items()},
    )


def _selected_band_indices(band_count: int, band_subset: Sequence[int] | None) -> list[int]:
    """Return the 1-based band indices to report, validating a requested subset."""
    if band_subset is None:
        return list(range(1, band_count + 1))
    out_of_range = sorted({index for index in band_subset if index < 1 or index > band_count})
    if out_of_range:
        raise ProductStructureError(
            "Requested band indices are outside the raster's band range.",
            hint=f"This raster has {band_count} band(s); indices are 1-based.",
            context={"requested": list(band_subset), "out_of_range": out_of_range},
        )
    return sorted(set(band_subset))


def inspect_raster(
    path: Path,
    *,
    band_subset: Sequence[int] | None = None,
    compute_checksum: bool = False,
) -> RasterInfo:
    """Inspect a raster's metadata without reading any pixels.

    Args:
        path: Raster file to open.
        band_subset: 1-based band indices to report. ``None`` reports every band.
        compute_checksum: Whether to hash the file, which is expensive on large products.

    Returns:
        A populated :class:`~hypersat.models.product.RasterInfo`.

    Raises:
        RasterReadError: If the file cannot be opened or its metadata cannot be read.
        ProductStructureError: If ``band_subset`` refers to non-existent bands.
    """
    collected_warnings: list[str] = []
    try:
        # A raster in sensor geometry has no geotransform, and rasterio warns about it on
        # open. That is expected for L1B input, so the warning is captured and reported as
        # a field instead of being printed as a Python warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", NotGeoreferencedWarning)
            with rasterio.open(path) as dataset:
                collected_warnings.extend(
                    str(item.message)
                    for item in caught
                    if issubclass(item.category, NotGeoreferencedWarning)
                )
                info = _raster_info_from_dataset(
                    dataset,
                    path=path,
                    band_subset=band_subset,
                    compute_checksum=compute_checksum,
                    collected_warnings=collected_warnings,
                )
    except RasterioIOError as error:
        raise RasterReadError(
            "Could not open the raster.",
            hint="Verify the file is a raster in a format GDAL supports and is not "
            "truncated. `hypersat inspect` requires a readable dataset, and for a product "
            "directory the imagery file itself can be passed with --input.",
            context={"path": str(path), "reason": str(error)},
        ) from error
    return info


def _raster_info_from_dataset(
    dataset: rasterio.DatasetReader,
    *,
    path: Path,
    band_subset: Sequence[int] | None,
    compute_checksum: bool,
    collected_warnings: list[str],
) -> RasterInfo:
    """Build :class:`RasterInfo` from an open dataset."""
    band_count = int(dataset.count)
    indices = _selected_band_indices(band_count, band_subset)
    bands = [_band_info(dataset, index) for index in indices]

    transform = dataset.transform
    transform_is_identity = bool(transform.is_identity)
    crs = _crs_info(dataset, collected_warnings)
    has_affine = crs.is_defined and not transform_is_identity

    bounds: tuple[float, float, float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    if not transform_is_identity:
        raster_bounds = dataset.bounds
        bounds = (
            float(raster_bounds.left),
            float(raster_bounds.bottom),
            float(raster_bounds.right),
            float(raster_bounds.top),
        )
        pixel_size = (abs(float(transform.a)), abs(float(transform.e)))

    nodata = _to_float(dataset.nodata)
    nodata_is_nan = nodata is not None and bool(np.isnan(nodata))
    itemsize = max(int(np.dtype(str(dtype)).itemsize) for dtype in dataset.dtypes)

    file_info: FileInfo = describe_file(path, compute_checksum=compute_checksum)
    sidecars = [Path(str(item)) for item in dataset.files[1:]]

    if not crs.is_defined:
        collected_warnings.append("Raster has no CRS.")
    if nodata is None:
        collected_warnings.append("Raster has no NoData value configured.")

    return RasterInfo(
        path=path,
        driver=str(dataset.driver),
        width=int(dataset.width),
        height=int(dataset.height),
        band_count=band_count,
        dtypes=[str(dtype) for dtype in dataset.dtypes],
        nodata=None if nodata_is_nan else nodata,
        nodata_is_nan=nodata_is_nan,
        nodata_per_band=[band.nodata for band in bands],
        crs=crs,
        transform=[float(value) for value in tuple(transform)[:6]],
        transform_is_identity=transform_is_identity,
        has_affine_georeferencing=has_affine,
        bounds=bounds,
        pixel_size=pixel_size,
        is_tiled=bool(dataset.is_tiled),
        block_shapes=[(int(shape[0]), int(shape[1])) for shape in dataset.block_shapes],
        compression=str(dataset.compression.name) if dataset.compression else None,
        interleaving=str(dataset.interleaving.name) if dataset.interleaving else None,
        metadata_domains=[str(domain) for domain in dataset.tag_namespaces()],
        metadata={str(key): str(value) for key, value in dataset.tags().items()},
        rpc=_rpc_info(dataset),
        gcp_count=len(dataset.gcps[0]),
        bands=bands,
        file=file_info,
        sidecar_files=sidecars,
        estimated_uncompressed_bytes=int(dataset.width)
        * int(dataset.height)
        * band_count
        * itemsize,
        warnings=collected_warnings,
    )


def inspect_input(
    input_path: Path,
    *,
    band_subset: Sequence[int] | None = None,
    compute_checksum: bool = False,
    proj_autofix: bool = True,
) -> InspectionResult:
    """Inspect a raster file or product directory and describe the runtime as well.

    Args:
        input_path: Raster file or product directory.
        band_subset: 1-based band indices to report. ``None`` reports every band.
        compute_checksum: Whether to hash the raster file.
        proj_autofix: Whether to fall back to rasterio's bundled PROJ database when the
            one configured through the environment is unusable.

    Returns:
        A populated :class:`~hypersat.models.product.InspectionResult`.

    Raises:
        ProductStructureError: If the input path or product structure is unusable.
        RasterReadError: If the raster cannot be opened.
    """
    proj_status: ProjDataStatus = ensure_usable_proj_data(allow_repair=proj_autofix)
    resolved, layout = resolve_raster_path(input_path)
    raster = inspect_raster(resolved, band_subset=band_subset, compute_checksum=compute_checksum)

    result_warnings: list[str] = []
    if proj_status is ProjDataStatus.REPAIRED:
        result_warnings.append(
            "The PROJ database configured in the environment was unusable; "
            "rasterio's bundled database was used instead."
        )
    elif proj_status is ProjDataStatus.BROKEN:
        result_warnings.append(
            "No usable PROJ database was found; CRS information may be incomplete."
        )

    logger.info(
        "inspected raster",
        extra={
            "raster": str(resolved),
            "width": raster.width,
            "height": raster.height,
            "bands": raster.band_count,
            "driver": raster.driver,
            "rpc_available": raster.rpc.available,
        },
    )
    return InspectionResult(
        input_path=input_path,
        input_kind=InputKind.DIRECTORY if layout is not None else InputKind.FILE,
        resolved_raster_path=resolved,
        raster=raster,
        product=layout,
        environment=describe_environment(proj_status),
        warnings=result_warnings,
    )
