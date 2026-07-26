"""Human-readable rendering of inspection and validation results.

Pure functions from model to string: no printing, no I/O, no colour codes. That keeps the
CLI handlers thin, makes the output testable with plain string assertions, and means the
same text can be reused in a log or a report later.

Machine-readable output is produced by Pydantic's JSON serialiser, not here.
"""

from __future__ import annotations

from hypersat.models.environment import EnvironmentInfo, ProjDataStatus
from hypersat.models.product import InspectionResult, RasterInfo
from hypersat.models.validation import CheckStatus, ValidationReport

__all__ = ["render_environment", "render_inspection", "render_validation_report"]

_LABEL_WIDTH = 24
_BYTES_PER_GIB = 1024**3
_STATUS_LABELS = {
    CheckStatus.PASSED: "PASS",
    CheckStatus.FAILED: "FAIL",
    CheckStatus.WARNING: "WARN",
    CheckStatus.SKIPPED: "SKIP",
}


def _row(label: str, value: object) -> str:
    """Render one indented ``label  value`` line."""
    return f"  {label:<{_LABEL_WIDTH}} {value}"


def _yes_no(value: bool) -> str:
    """Render a boolean as ``yes``/``no``."""
    return "yes" if value else "no"


def _format_nodata(nodata: float | None, is_nan: bool, *, unset: str) -> str:
    """Render a NoData value, distinguishing "NaN" from "not configured"."""
    if is_nan:
        return "nan"
    return unset if nodata is None else f"{nodata:g}"


def render_environment(environment: EnvironmentInfo) -> str:
    """Render the geospatial runtime as a single line."""
    parts = [
        f"rasterio {environment.rasterio_version}",
        f"GDAL {environment.gdal_version}",
        f"PROJ {environment.proj_version}",
        f"proj-data {environment.proj_data_status.value}",
    ]
    bindings = environment.gdal_bindings_version
    parts.append(f"osgeo.gdal {bindings}" if bindings else "osgeo.gdal not installed")
    return ", ".join(parts)


def _render_bands(raster: RasterInfo, band_limit: int) -> list[str]:
    """Render the per-band table, truncated to ``band_limit`` rows."""
    shown = raster.bands if band_limit <= 0 else raster.bands[:band_limit]
    if not shown:
        return []

    truncated = len(shown) < raster.band_count
    lines = [
        "",
        f"Bands ({len(shown)} of {raster.band_count} shown)"
        if truncated
        else f"Bands ({raster.band_count})",
        f"  {'idx':>4}  {'dtype':<9} {'nodata':>10}  {'wavelength':>12}  "
        f"{'source':<15} description",
    ]
    for band in shown:
        nodata = _format_nodata(band.nodata, band.nodata_is_nan, unset="-")
        wavelength = "-" if band.wavelength_nm is None else f"{band.wavelength_nm:.2f} nm"
        source = band.wavelength_source.value if band.wavelength_source else "-"
        lines.append(
            f"  {band.index:>4}  {band.dtype:<9} {nodata:>10}  {wavelength:>12}  "
            f"{source:<15} {band.description or '-'}"
        )
    if truncated:
        lines.append(
            f"  ... {raster.band_count - len(shown)} more band(s); "
            "use --json or --band-limit 0 for all"
        )
    return lines


def _render_input_section(result: InspectionResult) -> list[str]:
    """Render the input path and, for a product directory, its layout summary."""
    raster = result.raster
    lines = ["Input", _row("path", result.input_path), _row("kind", result.input_kind.value)]
    if result.product is not None:
        lines += [
            _row("raster", result.resolved_raster_path),
            _row("product size", result.product.total_size_human),
            _row("raster candidates", len(result.product.raster_candidates)),
            _row("metadata files", len(result.product.metadata_files)),
        ]
    lines.append(_row("file size", raster.file.size_human))
    if raster.file.sha256:
        lines.append(_row("sha256", raster.file.sha256))
    if raster.sidecar_files:
        lines.append(_row("sidecar files", ", ".join(path.name for path in raster.sidecar_files)))
    return lines


def _render_raster_section(raster: RasterInfo) -> list[str]:
    """Render driver, dimensions, band count, data types and NoData."""
    nodata = _format_nodata(raster.nodata, raster.nodata_is_nan, unset="not set")
    estimate = raster.estimated_uncompressed_bytes / _BYTES_PER_GIB
    return [
        "",
        "Raster",
        _row("driver", raster.driver),
        _row("dimensions", f"{raster.width} x {raster.height} px"),
        _row("bands", raster.band_count),
        _row("data types", ", ".join(sorted(set(raster.dtypes))) or "-"),
        _row("nodata", nodata),
        _row("uncompressed estimate", f"{estimate:.3f} GiB"),
    ]


def _render_georeferencing_section(raster: RasterInfo) -> list[str]:
    """Render CRS, transform, bounds, pixel size and sensor-model information."""
    crs_value = "not set"
    if raster.crs.is_defined:
        kind = "geographic" if raster.crs.is_geographic else "projected"
        units = f", {raster.crs.linear_units}" if raster.crs.linear_units else ""
        crs_value = f"{raster.crs.authority_code or 'defined (no EPSG code)'} ({kind}{units})"

    lines = [
        "",
        "Georeferencing",
        _row("crs", crs_value),
        _row("affine georeferencing", _yes_no(raster.has_affine_georeferencing)),
        _row("transform", ", ".join(f"{value:g}" for value in raster.transform) or "-"),
        _row(
            "bounds", ", ".join(f"{value:g}" for value in raster.bounds) if raster.bounds else "-"
        ),
        _row(
            "pixel size",
            f"{raster.pixel_size[0]:g} x {raster.pixel_size[1]:g}" if raster.pixel_size else "-",
        ),
    ]
    if raster.rpc.available:
        usability = "usable" if raster.rpc.is_usable else "NOT USABLE"
        lines.append(_row("rpc sensor model", f"available ({usability})"))
        lines.append(_row("rpc height offset", raster.rpc.height_off))
        lines.extend(_row("rpc issue", issue) for issue in raster.rpc.issues)
    else:
        lines.append(_row("rpc sensor model", "not available"))
    lines.append(_row("gcps", raster.gcp_count))
    return lines


def _render_storage_section(raster: RasterInfo) -> list[str]:
    """Render tiling, block shape, compression and interleaving."""
    block = (
        f"{raster.block_shapes[0][1]} x {raster.block_shapes[0][0]}" if raster.block_shapes else "-"
    )
    return [
        "",
        "Storage",
        _row("tiled", _yes_no(raster.is_tiled)),
        _row("block shape", block),
        _row("compression", raster.compression or "none"),
        _row("interleaving", raster.interleaving or "-"),
    ]


def _render_metadata_section(raster: RasterInfo) -> list[str]:
    """Render metadata domains, dataset tag count and wavelength coverage."""
    wavelengths = raster.wavelengths_nm
    coverage = (
        f"{len(wavelengths)} band(s), {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm"
        if wavelengths
        else "incomplete or absent"
    )
    return [
        "",
        "Metadata",
        _row("domains", ", ".join(raster.metadata_domains) or "none"),
        _row("dataset tags", len(raster.metadata)),
        _row("wavelength coverage", coverage),
    ]


def render_inspection(result: InspectionResult, *, band_limit: int = 10) -> str:
    """Render an inspection result as plain text.

    Args:
        result: The inspection result to render.
        band_limit: Maximum number of band rows to print; ``0`` prints every band.

    Returns:
        A multi-line string without a trailing newline.
    """
    raster = result.raster
    lines = [
        *_render_input_section(result),
        *_render_raster_section(raster),
        *_render_georeferencing_section(raster),
        *_render_storage_section(raster),
        *_render_metadata_section(raster),
        *_render_bands(raster, band_limit),
    ]

    all_warnings = [*result.warnings, *raster.warnings]
    if all_warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"  - {warning}" for warning in all_warnings)

    lines.append("")
    lines.append("Environment")
    lines.append(f"  {render_environment(result.environment)}")
    if result.environment.proj_data_status is not ProjDataStatus.OK:
        lines.append(f"  PROJ_LIB/PROJ_DATA: {result.environment.proj_lib_env or 'not set'}")
    return "\n".join(lines)


def render_validation_report(report: ValidationReport) -> str:
    """Render a validation report as plain text.

    Args:
        report: The report to render.

    Returns:
        A multi-line string without a trailing newline.
    """
    counts = report.counts()
    verdict = "VALID" if report.is_valid else "INVALID"
    lines = [
        f"Validation: {verdict} "
        f"({counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['warning']} warning, {counts['skipped']} skipped)",
        _row("input", report.input_path),
    ]
    if report.resolved_raster_path is not None:
        lines.append(_row("raster", report.resolved_raster_path))
    if report.treat_warnings_as_errors:
        lines.append(_row("strict mode", "warnings are treated as errors"))
    lines.append("")

    for check in report.checks:
        lines.append(f"  [{_STATUS_LABELS[check.status]}] {check.name:<22} {check.message}")
        if check.hint and check.status in (CheckStatus.FAILED, CheckStatus.WARNING):
            lines.append(f"         hint: {check.hint}")
    return "\n".join(lines)
