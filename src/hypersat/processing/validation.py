"""Pre-flight validation of products, elevation models and output locations.

The module offers two views of the same checks:

* **Fail-fast helpers** (:func:`validate_dem`, :func:`validate_output_directory`) raise a
  specific :class:`hypersat.exceptions.HyperSatError` subclass. Processing stages use
  these, because a stage that cannot proceed should stop immediately.
* **An aggregating pass** (:func:`validate_request`) runs every check and collects the
  findings into a :class:`~hypersat.models.validation.ValidationReport`. The CLI uses
  this, because an operator preparing a long run wants to see all problems at once.

Validation reads metadata only - no pixels - so it stays cheap enough to run before every
job.
"""

from __future__ import annotations

import uuid
from itertools import pairwise
from pathlib import Path

from hypersat.exceptions import (
    DependencyError,
    HyperSatError,
    InvalidWavelengthMetadataError,
    MissingDEMError,
    MissingGeoreferencingError,
    MissingRPCMetadataError,
    OutputPathError,
    ProductStructureError,
    ProductValidationError,
    RasterReadError,
    UnreadableDEMError,
)
from hypersat.io.environment import PROJ_CONFLICT_HINT, ensure_usable_proj_data
from hypersat.io.files import format_bytes
from hypersat.io.inspect import inspect_raster, resolve_raster_path
from hypersat.logging_config import get_logger
from hypersat.models.config import ValidationRequest
from hypersat.models.environment import ProjDataStatus
from hypersat.models.product import RasterInfo
from hypersat.models.validation import CheckStatus, ValidationCheck, ValidationReport

__all__ = [
    "BYTES_PER_GB",
    "validate_dem",
    "validate_output_directory",
    "validate_request",
]

logger = get_logger(__name__)

BYTES_PER_GB = 1024**3

_ERROR_TYPES: dict[str, type[HyperSatError]] = {
    error_type.__name__: error_type
    for error_type in (
        DependencyError,
        InvalidWavelengthMetadataError,
        MissingDEMError,
        MissingGeoreferencingError,
        MissingRPCMetadataError,
        OutputPathError,
        ProductStructureError,
        ProductValidationError,
        RasterReadError,
        UnreadableDEMError,
    )
}


def _check(
    name: str,
    status: CheckStatus,
    message: str,
    *,
    hint: str | None = None,
    error_type: type[HyperSatError] | None = None,
    **context: object,
) -> ValidationCheck:
    """Build a :class:`ValidationCheck`, keeping call sites terse and consistent."""
    return ValidationCheck(
        name=name,
        status=status,
        message=message,
        hint=hint,
        error_type=error_type.__name__ if error_type is not None else None,
        context=dict(context),
    )


def _from_error(name: str, error: HyperSatError) -> ValidationCheck:
    """Convert a raised domain error into a failed check, preserving hint and context."""
    return ValidationCheck(
        name=name,
        status=CheckStatus.FAILED,
        message=error.message,
        hint=error.hint,
        error_type=type(error).__name__,
        context=error.context,
    )


def validate_output_directory(directory: Path, *, overwrite: bool = False) -> None:
    """Ensure an output directory exists and is writable by this process.

    Creating a probe file is the only reliable portable test: on Windows, ACLs and
    read-only flags are not visible through ``os.access``.

    Args:
        directory: Directory that will receive output products.
        overwrite: Whether existing files may be replaced. Only used for the log message;
            individual stages enforce it per file.

    Raises:
        OutputPathError: If the directory cannot be created or written to.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputPathError(
            "Output directory could not be created.",
            hint="Choose a writable location with --output-dir, or create the directory "
            "manually and grant this user write permission.",
            context={"output_dir": str(directory), "reason": str(error)},
        ) from error

    if not directory.is_dir():
        raise OutputPathError(
            "Output path exists but is not a directory.",
            hint="Point --output-dir at a directory, not a file.",
            context={"output_dir": str(directory)},
        )

    probe = directory / f".hypersat-write-test-{uuid.uuid4().hex}"
    try:
        probe.write_bytes(b"")
    except OSError as error:
        raise OutputPathError(
            "Output directory is not writable.",
            hint="Grant write permission to this user, or choose another --output-dir.",
            context={"output_dir": str(directory), "reason": str(error)},
        ) from error
    finally:
        probe.unlink(missing_ok=True)

    logger.debug(
        "output directory is writable",
        extra={"output_dir": str(directory), "overwrite": overwrite},
    )


def validate_dem(dem_path: Path) -> RasterInfo:
    """Validate that a DEM exists, opens, and carries the georeferencing warping needs.

    A DEM is only usable for orthorectification if its heights can be located on the
    ground, which requires both a CRS and an affine transform. Whether it actually covers
    the scene footprint cannot be checked here: for a product in sensor geometry the
    footprint follows from evaluating the RPC model, which belongs to the
    orthorectification milestone.

    Args:
        dem_path: Path to the elevation model.

    Returns:
        The DEM's :class:`~hypersat.models.product.RasterInfo`.

    Raises:
        MissingDEMError: If the path does not exist or is not a file.
        UnreadableDEMError: If it cannot be opened or lacks usable georeferencing.
    """
    if not dem_path.exists():
        raise MissingDEMError(
            "DEM file does not exist.",
            hint="Download a DEM covering the scene (Copernicus DEM GLO-30 is a good "
            "default; see docs/data-sources.md) and pass it with --dem.",
            context={"dem_path": str(dem_path)},
        )
    if not dem_path.is_file():
        raise MissingDEMError(
            "DEM path is not a file.",
            hint="Pass a single DEM raster; a mosaic of tiles must be merged or exposed "
            "through a VRT first.",
            context={"dem_path": str(dem_path)},
        )

    try:
        info = inspect_raster(dem_path)
    except RasterReadError as error:
        raise UnreadableDEMError(
            "DEM exists but could not be opened.",
            hint="Verify the file is a raster GDAL can read and is not truncated.",
            context={"dem_path": str(dem_path), "reason": error.message},
        ) from error

    if not info.has_affine_georeferencing:
        raise UnreadableDEMError(
            "DEM has no CRS and affine transform, so its heights cannot be located.",
            hint="Use a DEM with proper georeferencing; a plain height array is not "
            "sufficient for orthorectification.",
            context={
                "dem_path": str(dem_path),
                "crs_defined": info.crs.is_defined,
                "transform_is_identity": info.transform_is_identity,
            },
        )
    return info


def _validate_raster_geometry(
    info: RasterInfo, request: ValidationRequest
) -> list[ValidationCheck]:
    """Run the dimension, band-count, band-subset and NoData checks."""
    checks: list[ValidationCheck] = []

    if info.width > 0 and info.height > 0:
        checks.append(
            _check(
                "raster_dimensions",
                CheckStatus.PASSED,
                f"Raster is {info.width} x {info.height} pixels.",
                width=info.width,
                height=info.height,
            )
        )
    else:
        checks.append(
            _check(
                "raster_dimensions",
                CheckStatus.FAILED,
                "Raster reports a zero width or height.",
                hint="The product is likely truncated; download it again.",
                error_type=ProductValidationError,
                width=info.width,
                height=info.height,
            )
        )

    if info.band_count > 0:
        checks.append(
            _check(
                "band_count",
                CheckStatus.PASSED,
                f"Raster has {info.band_count} band(s).",
                band_count=info.band_count,
            )
        )
    else:
        checks.append(
            _check(
                "band_count",
                CheckStatus.FAILED,
                "Raster contains no bands.",
                hint="The product is likely truncated or is a metadata-only file.",
                error_type=ProductValidationError,
                band_count=info.band_count,
            )
        )

    subset = request.product.band_subset
    if subset is None:
        checks.append(_check("band_subset", CheckStatus.SKIPPED, "No band subset requested."))
    else:
        out_of_range = sorted(index for index in subset if index > info.band_count)
        if out_of_range:
            checks.append(
                _check(
                    "band_subset",
                    CheckStatus.FAILED,
                    "Requested bands do not exist in this raster.",
                    hint=f"Band indices are 1-based and this raster has {info.band_count}.",
                    error_type=ProductValidationError,
                    requested=list(subset),
                    out_of_range=out_of_range,
                )
            )
        else:
            checks.append(
                _check(
                    "band_subset",
                    CheckStatus.PASSED,
                    f"All {len(subset)} requested band(s) exist.",
                    requested=list(subset),
                )
            )

    if info.nodata is not None or info.nodata_is_nan:
        checks.append(
            _check(
                "nodata_configured",
                CheckStatus.PASSED,
                "NaN NoData value is configured."
                if info.nodata_is_nan
                else f"NoData value is {info.nodata}.",
                nodata=info.nodata,
                nodata_is_nan=info.nodata_is_nan,
            )
        )
    else:
        checks.append(
            _check(
                "nodata_configured",
                CheckStatus.WARNING,
                "Raster has no NoData value, so pixels outside the swath cannot be "
                "distinguished from valid observations.",
                hint="Warping will fill new areas with 0, which is indistinguishable from "
                "real data. Set a NoData value on the product before orthorectification.",
            )
        )

    return checks


def _validate_georeferencing(info: RasterInfo, request: ValidationRequest) -> ValidationCheck:
    """Check that the raster can be placed on the ground at all.

    Evaluated *after* the RPC check: when a sensor model was required and is absent, both
    checks fail, and "no RPC sensor model" is the more specific and more actionable of the
    two, so it should be the error the process exits with.
    """
    has_rpc = info.rpc.available
    if info.has_affine_georeferencing or has_rpc:
        how = []
        if info.has_affine_georeferencing:
            how.append(f"affine transform with {info.crs.authority_code or 'a defined CRS'}")
        if has_rpc:
            how.append("RPC sensor model")
        return _check(
            "georeferencing",
            CheckStatus.PASSED,
            "Georeferencing present: " + " and ".join(how) + ".",
            has_affine_georeferencing=info.has_affine_georeferencing,
            rpc_available=has_rpc,
        )

    status = (
        CheckStatus.FAILED if request.requirements.require_georeferencing else CheckStatus.WARNING
    )
    return _check(
        "georeferencing",
        status,
        "Raster has neither a CRS with an affine transform nor an RPC sensor model.",
        hint="Only a product carrying georeferencing can be placed on a map grid. "
        "An EnMAP L1B product should expose RPC metadata; if it does not, the imagery file "
        "inside the product directory may be the wrong one.",
        error_type=MissingGeoreferencingError,
        crs_defined=info.crs.is_defined,
        transform_is_identity=info.transform_is_identity,
    )


def _validate_rpc(info: RasterInfo, request: ValidationRequest) -> ValidationCheck:
    """Check RPC availability and completeness against the request's requirements."""
    required = request.requirements.require_rpc
    if not info.rpc.available:
        if not required:
            return _check(
                "rpc_metadata",
                CheckStatus.SKIPPED,
                "No RPC sensor model present, and none required by this request.",
                hint="Orthorectification requires RPCs; re-run with --require-rpc to make "
                "this a hard failure.",
            )
        return _check(
            "rpc_metadata",
            CheckStatus.FAILED,
            "Orthorectification requires an RPC sensor model, but the raster has none.",
            hint="Use a product in sensor geometry that ships RPC metadata (EnMAP L1B). "
            "Reprojection is not a substitute: without a sensor model and a DEM, terrain "
            "displacement cannot be corrected. See docs/orthorectification.md. Note that "
            "GDAL discards an RPC set that is missing required keys, so a partially "
            "written model also appears as 'no RPC'.",
            error_type=MissingRPCMetadataError,
            rpc_available=False,
        )

    if not info.rpc.is_usable:
        status = CheckStatus.FAILED if required else CheckStatus.WARNING
        return _check(
            "rpc_metadata",
            status,
            "RPC metadata is present but not usable: " + "; ".join(info.rpc.issues) + ".",
            hint="A usable RPC00B model needs 20 non-degenerate coefficients in each of "
            "the four polynomials and non-zero normalisation scales. The product may be "
            "damaged; re-download it before attempting orthorectification.",
            error_type=MissingRPCMetadataError,
            issues=info.rpc.issues,
            coefficient_counts=info.rpc.coefficient_counts,
        )

    return _check(
        "rpc_metadata",
        CheckStatus.PASSED,
        "Usable RPC sensor model present.",
        height_off=info.rpc.height_off,
        err_bias=info.rpc.err_bias,
    )


def _validate_wavelengths(info: RasterInfo, request: ValidationRequest) -> ValidationCheck:
    """Check per-band centre wavelengths, or the operator-supplied override."""
    override = request.product.wavelengths_nm
    if override is not None:
        if len(override) != info.band_count:
            return _check(
                "wavelength_metadata",
                CheckStatus.FAILED,
                "The supplied wavelength override does not match the band count.",
                hint=f"Provide exactly {info.band_count} values, or remove the override to "
                "read wavelengths from the product metadata.",
                error_type=InvalidWavelengthMetadataError,
                supplied=len(override),
                band_count=info.band_count,
            )
        return _check(
            "wavelength_metadata",
            CheckStatus.PASSED,
            f"Using {len(override)} operator-supplied centre wavelengths.",
            source="config_override",
        )

    wavelengths = info.wavelengths_nm
    if wavelengths is None:
        missing = [band.index for band in info.bands if band.wavelength_nm is None]
        status = (
            CheckStatus.FAILED if request.requirements.require_wavelengths else CheckStatus.WARNING
        )
        return _check(
            "wavelength_metadata",
            status,
            f"{len(missing)} of {len(info.bands)} inspected band(s) have no centre "
            "wavelength in their metadata.",
            hint="Wavelength-based band selection needs centre wavelengths. Supply them "
            "explicitly with the wavelengths_nm configuration key, or use a product whose "
            "metadata includes them.",
            error_type=InvalidWavelengthMetadataError,
            missing_band_indices=missing[:20],
            missing_count=len(missing),
        )

    unsorted_pairs = sum(1 for earlier, later in pairwise(wavelengths) if later <= earlier)
    if unsorted_pairs:
        return _check(
            "wavelength_metadata",
            CheckStatus.WARNING,
            f"Centre wavelengths are not strictly increasing ({unsorted_pairs} step(s) "
            "descend or repeat).",
            hint="This is normal where a VNIR and a SWIR detector overlap. Nearest-band "
            "selection still works, but verify which detector a selected band came from.",
            band_count=len(wavelengths),
            first_nm=wavelengths[0],
            last_nm=wavelengths[-1],
        )

    return _check(
        "wavelength_metadata",
        CheckStatus.PASSED,
        f"All {len(wavelengths)} inspected band(s) have centre wavelengths "
        f"({wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm).",
        first_nm=wavelengths[0],
        last_nm=wavelengths[-1],
    )


def _validate_size(info: RasterInfo, request: ValidationRequest) -> ValidationCheck:
    """Guard against opening a product far larger than the machine can handle."""
    limit_gb = request.requirements.max_uncompressed_gb
    estimated_gb = info.estimated_uncompressed_bytes / BYTES_PER_GB
    if limit_gb is None:
        return _check(
            "product_size",
            CheckStatus.SKIPPED,
            "No uncompressed-size limit configured.",
            estimated_uncompressed=format_bytes(info.estimated_uncompressed_bytes),
        )
    if estimated_gb > limit_gb:
        return _check(
            "product_size",
            CheckStatus.FAILED,
            f"A full read of this raster would need about {estimated_gb:.2f} GiB, "
            f"above the configured limit of {limit_gb:.2f} GiB.",
            hint="Raise max_uncompressed_gb if the machine can take it, or restrict "
            "processing with a band subset. Reading is windowed, so this is a guard "
            "against naive full-cube operations rather than a hard technical limit.",
            error_type=ProductValidationError,
            estimated_gb=round(estimated_gb, 3),
            limit_gb=limit_gb,
        )
    return _check(
        "product_size",
        CheckStatus.PASSED,
        f"A full read would need about {estimated_gb:.2f} GiB, within the "
        f"{limit_gb:.2f} GiB limit.",
        estimated_gb=round(estimated_gb, 3),
        limit_gb=limit_gb,
    )


def _validate_proj_database(request: ValidationRequest) -> ValidationCheck:
    """Check that the PROJ coordinate database is usable before any CRS work starts."""
    status = ensure_usable_proj_data(allow_repair=request.proj_autofix)
    if status is ProjDataStatus.OK:
        return _check("proj_database", CheckStatus.PASSED, "PROJ database is usable.")
    if status is ProjDataStatus.REPAIRED:
        return _check(
            "proj_database",
            CheckStatus.WARNING,
            "The PROJ database configured in the environment is unusable; rasterio's "
            "bundled database was used instead.",
            hint=PROJ_CONFLICT_HINT,
            proj_data_status=status.value,
        )
    return _check(
        "proj_database",
        CheckStatus.FAILED,
        "No usable PROJ database was found, so coordinate reference systems cannot be interpreted.",
        hint=PROJ_CONFLICT_HINT,
        error_type=DependencyError,
        proj_data_status=status.value,
    )


def _validate_dem_checks(request: ValidationRequest) -> list[ValidationCheck]:
    """Validate the DEM when one was supplied, reporting each finding separately."""
    if request.dem_path is None:
        message = "No DEM supplied."
        hint = (
            "Orthorectification requires a DEM; pass one with --dem. See docs/data-sources.md."
            if request.requirements.require_rpc
            else None
        )
        return [_check("dem", CheckStatus.SKIPPED, message, hint=hint)]

    try:
        dem_info = validate_dem(request.dem_path)
    except (MissingDEMError, UnreadableDEMError) as error:
        return [_from_error("dem", error)]

    checks = [
        _check(
            "dem",
            CheckStatus.PASSED,
            f"DEM is readable, {dem_info.width} x {dem_info.height} pixels in "
            f"{dem_info.crs.authority_code or 'a defined CRS'}.",
            dem_path=str(request.dem_path),
            crs=dem_info.crs.authority_code,
            pixel_size=list(dem_info.pixel_size) if dem_info.pixel_size else None,
        )
    ]

    if dem_info.band_count != 1:
        checks.append(
            _check(
                "dem_band_count",
                CheckStatus.WARNING,
                f"DEM has {dem_info.band_count} bands; a DEM is normally single-band.",
                hint="Warping will use the first band as elevation. Verify that is the "
                "height band.",
                band_count=dem_info.band_count,
            )
        )
    if dem_info.nodata is None and not dem_info.nodata_is_nan:
        checks.append(
            _check(
                "dem_nodata",
                CheckStatus.WARNING,
                "DEM has no NoData value, so voids cannot be recognised.",
                hint="Void pixels would be treated as real elevations, displacing the "
                "affected image area. Set the DEM's NoData value, or configure "
                "RPC_DEM_MISSING_VALUE.",
            )
        )
    return checks


def validate_request(request: ValidationRequest) -> ValidationReport:
    """Run every applicable pre-flight check and collect the findings.

    Checks stop early only when there is nothing left to inspect - an unresolvable input
    path or an unreadable raster. Everything else is evaluated so the operator sees all
    problems in one pass.

    Args:
        request: What to validate and which requirements apply.

    Returns:
        A :class:`~hypersat.models.validation.ValidationReport`.
    """
    input_path = request.product.path
    checks: list[ValidationCheck] = [_validate_proj_database(request)]

    try:
        resolved, layout = resolve_raster_path(input_path)
    except ProductStructureError as error:
        checks.append(_from_error("product_structure", error))
        return ValidationReport(
            input_path=input_path,
            checks=checks,
            treat_warnings_as_errors=request.treat_warnings_as_errors,
        )

    checks.append(
        _check(
            "product_structure",
            CheckStatus.PASSED,
            f"Resolved imagery to {resolved.name}."
            if layout is not None
            else "Input is a raster file.",
            resolved_raster=str(resolved),
            raster_candidates=len(layout.raster_candidates) if layout else 1,
            metadata_files=len(layout.metadata_files) if layout else 0,
        )
    )

    try:
        info = inspect_raster(resolved)
    except RasterReadError as error:
        checks.append(_from_error("raster_readable", error))
        return ValidationReport(
            input_path=input_path,
            resolved_raster_path=resolved,
            checks=checks,
            treat_warnings_as_errors=request.treat_warnings_as_errors,
        )

    checks.append(
        _check(
            "raster_readable",
            CheckStatus.PASSED,
            f"Raster opened with the {info.driver} driver.",
            driver=info.driver,
            file_size=info.file.size_human,
        )
    )
    checks.extend(_validate_raster_geometry(info, request))
    checks.append(_validate_rpc(info, request))
    checks.append(_validate_georeferencing(info, request))
    checks.append(_validate_wavelengths(info, request))
    checks.append(_validate_size(info, request))
    checks.extend(_validate_dem_checks(request))

    if request.output is None:
        checks.append(
            _check("output_directory", CheckStatus.SKIPPED, "No output directory supplied.")
        )
    else:
        try:
            validate_output_directory(request.output.directory, overwrite=request.output.overwrite)
        except OutputPathError as error:
            checks.append(_from_error("output_directory", error))
        else:
            existing = [item for item in request.output.directory.iterdir() if item.is_file()]
            if existing and not request.output.overwrite:
                checks.append(
                    _check(
                        "output_directory",
                        CheckStatus.WARNING,
                        f"Output directory already contains {len(existing)} file(s) and "
                        "overwrite is disabled.",
                        hint="Stages will refuse to replace existing products. Use a fresh "
                        "directory or enable overwrite.",
                        output_dir=str(request.output.directory),
                        existing_files=len(existing),
                    )
                )
            else:
                checks.append(
                    _check(
                        "output_directory",
                        CheckStatus.PASSED,
                        "Output directory exists and is writable.",
                        output_dir=str(request.output.directory),
                    )
                )

    report = ValidationReport(
        input_path=input_path,
        resolved_raster_path=resolved,
        checks=checks,
        treat_warnings_as_errors=request.treat_warnings_as_errors,
    )
    logger.info(
        "validation completed",
        extra={
            "input_path": str(input_path),
            "is_valid": report.is_valid,
            **report.counts(),
        },
    )
    return report


def raise_if_invalid(report: ValidationReport) -> None:
    """Raise the first blocking finding as its specific exception type.

    Args:
        report: A report produced by :func:`validate_request`.

    Raises:
        ProductValidationError: Or a more specific subclass, matching the failed check.
    """
    if report.is_valid:
        return

    blocking = report.blocking
    first = blocking[0]
    error_type = _ERROR_TYPES.get(first.error_type or "", ProductValidationError)
    context = dict(first.context)
    context["check"] = first.name
    if len(blocking) > 1:
        context["other_blocking_checks"] = [check.name for check in blocking[1:]]
    raise error_type(first.message, hint=first.hint, context=context)
