"""Command-line interface for the HyperSat pipeline.

The CLI is deliberately thin: a command parses options, builds a validated configuration
model, calls one service function and renders the result. Domain errors derive from
:class:`hypersat.exceptions.HyperSatError` and are translated into documented process exit
codes by :func:`main`.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from pydantic import BaseModel, ValidationError

from hypersat import __version__
from hypersat.exceptions import ConfigurationError, HyperSatError
from hypersat.formatting import render_environment, render_inspection, render_validation_report
from hypersat.io.environment import describe_environment
from hypersat.io.inspect import inspect_input
from hypersat.logging_config import LogFormat, configure_logging, get_logger
from hypersat.models.config import (
    DataSemantics,
    IndexRequest,
    InputConfig,
    MorphologyConfig,
    MorphologyKernelShape,
    MorphologyOperation,
    OrthorectifyRequest,
    OutputConfig,
    PreviewComposite,
    PreviewRequest,
    QualityMaskRequest,
    ReprojectRequest,
    ResamplingMethod,
    RpcTransformerOptions,
    SpectralIndexName,
    SpectralProfileRequest,
    StretchConfig,
    ValidationRequest,
    ValidationRequirements,
)
from hypersat.processing.orthorectification import orthorectify_raster
from hypersat.processing.quality_mask import build_quality_mask
from hypersat.processing.reprojection import reproject_raster
from hypersat.processing.spectral import calculate_index, extract_spectral_profile
from hypersat.processing.validation import raise_if_invalid, validate_request
from hypersat.visualization.preview import render_preview

logger = get_logger(__name__)

# PEP 695 generics would be nicer, but the package supports Python 3.11.
ModelT = TypeVar("ModelT", bound=BaseModel)

app = typer.Typer(
    name="hypersat",
    help=(
        "Demonstration pipeline for hyperspectral satellite imagery: inspection, "
        "validation, RPC/DEM orthorectification, spectral indices and QC reporting. "
        "Outputs are demonstration products, not official mission products."
    ),
    no_args_is_help=True,
    add_completion=False,
)

InputOption = Annotated[
    Path,
    typer.Option(
        "--input",
        "-i",
        help="Path to a raster file or a satellite-product directory.",
        show_default=False,
    ),
]
JsonFlag = Annotated[
    bool,
    typer.Option("--json", help="Emit machine-readable JSON on stdout instead of text."),
]
BandsOption = Annotated[
    str | None,
    typer.Option(
        "--bands",
        help="Comma-separated 1-based band indices to restrict the report to, e.g. '1,42,120'.",
        show_default=False,
    ),
]
ProjAutofixOption = Annotated[
    bool,
    typer.Option(
        "--proj-autofix/--no-proj-autofix",
        help="If the PROJ database configured in the environment is unusable, fall back to "
        "the one bundled with rasterio. The fallback is always reported.",
        envvar="HYPERSAT_PROJ_AUTOFIX",
    ),
]


def _parse_band_indices(raw: str | None) -> tuple[int, ...] | None:
    """Parse a ``--bands`` value into 1-based indices.

    Args:
        raw: Comma-separated indices, or ``None``.

    Returns:
        A tuple of indices, or ``None`` when no subset was requested.

    Raises:
        typer.BadParameter: If the value is not a comma-separated list of integers.
    """
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise typer.BadParameter("No band indices given.", param_hint="--bands")
    try:
        return tuple(int(token) for token in tokens)
    except ValueError as error:
        raise typer.BadParameter(
            f"Band indices must be integers; got {raw!r}.", param_hint="--bands"
        ) from error


def _build_model(factory: type[ModelT], **kwargs: object) -> ModelT:
    """Instantiate a configuration model, reporting problems as a configuration error.

    Pydantic's own error text is precise but noisy; wrapping it keeps CLI failures within
    the documented exception hierarchy and exit codes.

    Args:
        factory: The model class to build.
        **kwargs: Field values.

    Returns:
        The validated model instance.

    Raises:
        ConfigurationError: If validation fails.
    """
    try:
        return factory(**kwargs)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or factory.__name__}: {item['msg']}"
            for item in error.errors()
        )
        raise ConfigurationError(
            f"Invalid configuration: {problems}",
            hint="Check the option values against configs/pipeline.example.yaml.",
            context={"model": factory.__name__},
        ) from error


@app.callback()
def configure(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging verbosity: DEBUG, INFO, WARNING, ERROR or CRITICAL.",
            envvar="HYPERSAT_LOG_LEVEL",
        ),
    ] = "INFO",
    log_format: Annotated[
        LogFormat,
        typer.Option(
            "--log-format",
            help="Log rendering: human-readable text or newline-delimited JSON.",
            envvar="HYPERSAT_LOG_FORMAT",
        ),
    ] = LogFormat.TEXT,
) -> None:
    """Configure logging for every subcommand."""
    resolved = logging.getLevelNamesMapping().get(log_level.upper())
    if resolved is None:
        raise typer.BadParameter(f"Unknown log level {log_level!r}.", param_hint="--log-level")
    configure_logging(level=resolved, log_format=log_format)


@app.command()
def version(as_json: JsonFlag = False, verbose: bool = False) -> None:
    """Print the pipeline version, and with --verbose the geospatial runtime as well."""
    details: dict[str, object] = {
        "name": "hypersat-processing-pipeline",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if verbose:
        details["environment"] = describe_environment().to_json_dict()
    if as_json:
        typer.echo(json.dumps(details, indent=2))
        return
    typer.echo(f"hypersat {__version__} (Python {details['python']}, {details['platform']})")
    if verbose:
        environment = describe_environment()
        typer.echo(render_environment(environment))


@app.command()
def inspect(
    input_path: InputOption,
    as_json: JsonFlag = False,
    bands: BandsOption = None,
    checksum: Annotated[
        bool,
        typer.Option(
            "--checksum",
            help="Also compute the SHA-256 digest of the raster. Slow on large products.",
        ),
    ] = False,
    band_limit: Annotated[
        int,
        typer.Option(
            "--band-limit",
            min=0,
            help="Maximum number of band rows to print; 0 prints every band.",
        ),
    ] = 10,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Report dimensions, dtype, NoData, CRS, transform, bounds, RPC and wavelengths.

    Reads metadata only - no pixels - so it is safe to run on a multi-gigabyte
    hyperspectral cube. A product directory is scanned for its imagery file.
    """
    product = _build_model(
        InputConfig,
        path=input_path,
        band_subset=_parse_band_indices(bands),
    )
    result = inspect_input(
        product.path,
        band_subset=product.band_subset,
        compute_checksum=checksum,
        proj_autofix=proj_autofix,
    )
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(render_inspection(result, band_limit=band_limit))


@app.command()
def validate(
    input_path: InputOption,
    dem: Annotated[
        Path | None,
        typer.Option(
            "--dem",
            help="Digital elevation model to validate. Required for orthorectification.",
            show_default=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Output directory to check for existence and writability.",
            show_default=False,
        ),
    ] = None,
    bands: BandsOption = None,
    require_rpc: Annotated[
        bool,
        typer.Option(
            "--require-rpc/--no-require-rpc",
            help="Fail when the product carries no complete RPC sensor model.",
        ),
    ] = False,
    require_wavelengths: Annotated[
        bool,
        typer.Option(
            "--require-wavelengths/--no-require-wavelengths",
            help="Fail when any band lacks a centre wavelength.",
        ),
    ] = False,
    require_georeferencing: Annotated[
        bool,
        typer.Option(
            "--require-georeferencing/--no-require-georeferencing",
            help="Fail when the product has neither affine georeferencing nor an RPC model.",
        ),
    ] = True,
    max_uncompressed_gb: Annotated[
        float,
        typer.Option(
            "--max-uncompressed-gb",
            min=0.0,
            help="Fail when a full read of the raster would exceed this size. 0 disables.",
        ),
    ] = 16.0,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Allow existing files in the output directory."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Treat warnings as errors, for use in CI."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Run pre-flight checks on a product, a DEM and an output directory.

    Every check runs, so all problems are reported in one pass; the process then exits with
    the code of the first blocking finding.
    """
    request = _build_model(
        ValidationRequest,
        product=_build_model(InputConfig, path=input_path, band_subset=_parse_band_indices(bands)),
        requirements=_build_model(
            ValidationRequirements,
            require_georeferencing=require_georeferencing,
            require_rpc=require_rpc,
            require_wavelengths=require_wavelengths,
            max_uncompressed_gb=max_uncompressed_gb if max_uncompressed_gb > 0 else None,
        ),
        dem_path=dem,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite)
        if output_dir is not None
        else None,
        treat_warnings_as_errors=strict,
        proj_autofix=proj_autofix,
    )

    report = validate_request(request)
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(render_validation_report(report))
    raise_if_invalid(report)


@app.command()
def preview(
    input_path: InputOption,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory that receives the PNG preview.",
            show_default=False,
        ),
    ],
    composite: Annotated[
        PreviewComposite,
        typer.Option(
            "--composite",
            help="Composite to render: true-color, false-color, or band.",
        ),
    ] = PreviewComposite.TRUE_COLOR,
    band: Annotated[
        int | None,
        typer.Option(
            "--band",
            min=1,
            help="1-based band index for --composite band.",
            show_default=False,
        ),
    ] = None,
    bands: BandsOption = None,
    lower_percentile: Annotated[
        float,
        typer.Option("--lower-percentile", min=0.0, max=99.9, help="Lower stretch percentile."),
    ] = 2.0,
    upper_percentile: Annotated[
        float,
        typer.Option("--upper-percentile", min=0.1, max=100.0, help="Upper stretch percentile."),
    ] = 98.0,
    per_band: Annotated[
        bool,
        typer.Option(
            "--per-band/--joint",
            help="Stretch each band independently, or use one shared range across bands.",
        ),
    ] = True,
    max_dimension: Annotated[
        int,
        typer.Option(
            "--max-dimension",
            min=1,
            help="Longest preview side in pixels; larger inputs are downsampled while reading.",
        ),
    ] = 2048,
    blur_kernel: Annotated[
        int | None,
        typer.Option(
            "--blur-kernel",
            min=1,
            help="Odd Gaussian kernel size in pixels. Omit to skip blur.",
            show_default=False,
        ),
    ] = None,
    product_id: Annotated[
        str | None,
        typer.Option(
            "--product-id",
            help="Filename token; derived from the input path when omitted.",
            show_default=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing preview PNG."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Write a percentile-stretched PNG preview. Never modifies scientific rasters.

    Bands for true-color and false-color composites are chosen by wavelength when the
    product carries them; pass --bands to override. Large rasters are downsampled to
    --max-dimension while reading, so a hyperspectral cube is not loaded at full size.
    """
    request = _build_model(
        PreviewRequest,
        product_path=input_path,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite),
        composite=composite,
        bands=_parse_band_indices(bands),
        band=band,
        stretch=_build_model(
            StretchConfig,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            per_band=per_band,
        ),
        max_dimension=max_dimension,
        blur_kernel=blur_kernel,
        product_id=product_id,
        proj_autofix=proj_autofix,
    )
    result = render_preview(request)
    payload = {
        "path": str(result.path),
        "composite": result.composite.value,
        "band_indices": list(result.band_indices),
        "wavelengths_nm": list(result.wavelengths_nm),
        "width": result.width,
        "height": result.height,
        "product_id": result.product_id,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"wrote {result.path} ({result.width}x{result.height} px, "
        f"bands {','.join(str(index) for index in result.band_indices)})"
    )


_INDEX_BAND_COUNT = 2


def _parse_index_band_pair(raw: str | None) -> tuple[int, int] | None:
    """Parse ``--bands`` for an index command into a minuend/subtrahend pair."""
    indices = _parse_band_indices(raw)
    if indices is None:
        return None
    if len(indices) != _INDEX_BAND_COUNT:
        raise typer.BadParameter(
            "Spectral indices need exactly two band indices (minuend,subtrahend).",
            param_hint="--bands",
        )
    return indices[0], indices[1]


@app.command("calculate-index")
def calculate_index_command(
    input_path: InputOption,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory that receives the index GeoTIFF.",
            show_default=False,
        ),
    ],
    index: Annotated[
        SpectralIndexName,
        typer.Option("--index", help="Index to compute: ndvi or ndwi."),
    ] = SpectralIndexName.NDVI,
    bands: BandsOption = None,
    red_nm: Annotated[
        float,
        typer.Option("--red-nm", help="Target red wavelength in nanometres (NDVI)."),
    ] = 665.0,
    nir_nm: Annotated[
        float,
        typer.Option("--nir-nm", help="Target near-infrared wavelength in nanometres."),
    ] = 842.0,
    green_nm: Annotated[
        float,
        typer.Option("--green-nm", help="Target green wavelength in nanometres (NDWI)."),
    ] = 560.0,
    tolerance_nm: Annotated[
        float,
        typer.Option(
            "--tolerance-nm",
            min=0.0,
            help="Maximum accepted distance from each target wavelength.",
        ),
    ] = 15.0,
    nodata: Annotated[
        float,
        typer.Option("--nodata", help="NoData value written into the float32 GeoTIFF."),
    ] = -9999.0,
    statistics: Annotated[
        bool,
        typer.Option(
            "--statistics/--no-statistics",
            help="Also write a JSON summary of the index raster's valid pixels.",
        ),
    ] = False,
    product_id: Annotated[
        str | None,
        typer.Option(
            "--product-id",
            help="Filename token; derived from the input path when omitted.",
            show_default=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing index GeoTIFF."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Compute NDVI or NDWI and write a float32 GeoTIFF.

    Bands are chosen by wavelength when the product carries them. These are conventional
    broadband indices applied to two selected hyperspectral bands - not hyperspectral
    algorithms in their own right.
    """
    request = _build_model(
        IndexRequest,
        product_path=input_path,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite),
        index=index,
        bands=_parse_index_band_pair(bands),
        red_nm=red_nm,
        nir_nm=nir_nm,
        green_nm=green_nm,
        tolerance_nm=tolerance_nm,
        output_nodata=nodata,
        include_statistics=statistics,
        product_id=product_id,
        proj_autofix=proj_autofix,
    )
    result = calculate_index(request)
    payload = {
        "path": str(result.path),
        "index": result.index.value,
        "band_indices": list(result.band_indices),
        "wavelengths_nm": list(result.wavelengths_nm),
        "nodata": result.nodata,
        "product_id": result.product_id,
        "statistics_path": None if result.statistics_path is None else str(result.statistics_path),
        "statistics": None if result.statistics is None else result.statistics.to_dict(),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"wrote {result.path} ({result.index.value}, "
        f"bands {result.band_indices[0]},{result.band_indices[1]})"
    )
    if result.statistics_path is not None:
        typer.echo(f"wrote {result.statistics_path}")


@app.command("spectral-profile")
def spectral_profile_command(
    input_path: InputOption,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory that receives the CSV and JSON profile files.",
            show_default=False,
        ),
    ],
    row: Annotated[
        int,
        typer.Option("--row", min=0, help="Zero-based row of the sample pixel."),
    ],
    col: Annotated[
        int,
        typer.Option("--col", min=0, help="Zero-based column of the sample pixel."),
    ],
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            min=1,
            help="Odd neighbourhood size; 1 is a single pixel, larger averages neighbours.",
        ),
    ] = 1,
    bands: BandsOption = None,
    product_id: Annotated[
        str | None,
        typer.Option(
            "--product-id",
            help="Filename token; derived from the input path when omitted.",
            show_default=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing profile outputs."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Extract the spectrum at one pixel and write CSV + JSON.

    Coordinates are pixel indices, not map coordinates. Use --window-size with an odd
    value to average a small neighbourhood around the centre pixel.
    """
    request = _build_model(
        SpectralProfileRequest,
        product_path=input_path,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite),
        row=row,
        col=col,
        window_size=window_size,
        bands=_parse_band_indices(bands),
        product_id=product_id,
        proj_autofix=proj_autofix,
    )
    result = extract_spectral_profile(request)
    payload = {
        "product_id": result.product_id,
        "row": result.profile.row,
        "col": result.profile.col,
        "window_size": result.profile.window_size,
        "band_indices": list(result.profile.band_indices),
        "wavelengths_nm": list(result.profile.wavelengths_nm),
        "values": list(result.profile.values),
        "csv_path": str(result.csv_path),
        "json_path": str(result.json_path),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"wrote {result.csv_path} and {result.json_path} "
        f"(row={result.profile.row}, col={result.profile.col}, "
        f"{len(result.profile.values)} band(s))"
    )


def _parse_wavelengths(raw: str | None) -> tuple[float, ...] | None:
    """Parse a comma-separated wavelength list.

    An empty string means "use every band" (``None`` on the request model).
    """
    if raw is None:
        return None
    if not raw.strip():
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise typer.BadParameter("No wavelengths given.", param_hint="--evaluation-wavelengths-nm")
    try:
        return tuple(float(token) for token in tokens)
    except ValueError as error:
        raise typer.BadParameter(
            f"Wavelengths must be numbers; got {raw!r}.",
            param_hint="--evaluation-wavelengths-nm",
        ) from error


@app.command("quality-mask")
def quality_mask_command(
    input_path: InputOption,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory that receives the uint8 quality-mask GeoTIFF.",
            show_default=False,
        ),
    ],
    saturation_dn: Annotated[
        float,
        typer.Option("--saturation-dn", help="DN at or above this value is saturated."),
    ] = 65535.0,
    low_signal_dn: Annotated[
        float,
        typer.Option("--low-signal-dn", help="DN at or below this value is low signal."),
    ] = 10.0,
    band_fraction: Annotated[
        float,
        typer.Option(
            "--band-fraction",
            min=0.0,
            max=1.0,
            help="Fraction of evaluation bands that must meet a threshold.",
        ),
    ] = 0.5,
    evaluation_wavelengths_nm: Annotated[
        str | None,
        typer.Option(
            "--evaluation-wavelengths-nm",
            help="Comma-separated target wavelengths, e.g. '490,560,665,842'. "
            "Pass an empty string to evaluate every band. Ignored when --bands is set.",
        ),
    ] = "490,560,665,842",
    bands: BandsOption = None,
    morphology: Annotated[
        bool,
        typer.Option(
            "--morphology/--no-morphology",
            help="Apply morphological post-processing to defect classes.",
        ),
    ] = False,
    morphology_operation: Annotated[
        MorphologyOperation,
        typer.Option("--morphology-operation", help="open | close | dilate | erode | none."),
    ] = MorphologyOperation.CLOSE,
    morphology_kernel_shape: Annotated[
        MorphologyKernelShape,
        typer.Option("--morphology-kernel-shape", help="rect | ellipse | cross."),
    ] = MorphologyKernelShape.ELLIPSE,
    morphology_kernel_size: Annotated[
        int,
        typer.Option("--morphology-kernel-size", min=1, help="Odd kernel size in pixels."),
    ] = 3,
    morphology_iterations: Annotated[
        int,
        typer.Option("--morphology-iterations", min=1, help="Morphology iterations."),
    ] = 1,
    spectral_anomaly: Annotated[
        bool,
        typer.Option(
            "--spectral-anomaly/--no-spectral-anomaly",
            help="Enable the optional high-CV spectral anomaly class.",
        ),
    ] = False,
    anomaly_cv_threshold: Annotated[
        float,
        typer.Option(
            "--anomaly-cv-threshold",
            help="Coefficient-of-variation threshold for --spectral-anomaly.",
        ),
    ] = 2.0,
    statistics: Annotated[
        bool,
        typer.Option(
            "--statistics/--no-statistics",
            help="Also write a JSON summary of per-class counts and percentages.",
        ),
    ] = False,
    product_id: Annotated[
        str | None,
        typer.Option(
            "--product-id",
            help="Filename token; derived from the input path when omitted.",
            show_default=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing quality-mask GeoTIFF."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Build a uint8 quality mask in sensor geometry.

    Class codes follow docs/quality-masks.md. Saturation and low-signal thresholds are
    product-specific configuration, not mission-validated constants. Morphology is off by
    default and, when enabled, grows defect classes only.
    """
    request = _build_model(
        QualityMaskRequest,
        product_path=input_path,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite),
        saturation_dn=saturation_dn,
        low_signal_dn=low_signal_dn,
        saturation_band_fraction=band_fraction,
        evaluation_wavelengths_nm=_parse_wavelengths(evaluation_wavelengths_nm),
        bands=_parse_band_indices(bands),
        morphology=_build_model(
            MorphologyConfig,
            enabled=morphology,
            operation=morphology_operation,
            kernel_shape=morphology_kernel_shape,
            kernel_size=morphology_kernel_size,
            iterations=morphology_iterations,
        ),
        spectral_anomaly=spectral_anomaly,
        anomaly_cv_threshold=anomaly_cv_threshold,
        include_statistics=statistics,
        product_id=product_id,
        proj_autofix=proj_autofix,
    )
    result = build_quality_mask(request)
    payload = {
        "path": str(result.path),
        "band_indices": list(result.band_indices),
        "wavelengths_nm": list(result.wavelengths_nm),
        "counts": dict(result.counts),
        "product_id": result.product_id,
        "statistics_path": None if result.statistics_path is None else str(result.statistics_path),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"wrote {result.path} "
        f"(valid={result.counts.get('valid', 0)}, "
        f"nodata={result.counts.get('no_data', 0)}, "
        f"saturated={result.counts.get('saturated', 0)})"
    )
    if result.statistics_path is not None:
        typer.echo(f"wrote {result.statistics_path}")


@app.command("reproject")
def reproject_command(
    input_path: InputOption,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory that receives the reprojected GeoTIFF.",
            show_default=False,
        ),
    ],
    target_crs: Annotated[
        str,
        typer.Option(
            "--target-crs",
            help="Destination CRS: 'auto' (UTM of scene centre) or an authority code "
            "such as EPSG:32633. With --reference-raster, 'auto' inherits the reference CRS.",
        ),
    ] = "auto",
    resolution: Annotated[
        float | None,
        typer.Option(
            "--resolution",
            help="Output ground sample distance in target-CRS units. Omit to let GDAL "
            "choose from the source pixel size.",
            show_default=False,
        ),
    ] = None,
    resampling: Annotated[
        ResamplingMethod,
        typer.Option("--resampling", help="nearest | bilinear | cubic."),
    ] = ResamplingMethod.BILINEAR,
    data_semantics: Annotated[
        DataSemantics,
        typer.Option(
            "--data-semantics",
            help="continuous (default) or categorical. Categorical forces nearest.",
        ),
    ] = DataSemantics.CONTINUOUS,
    reference_raster: Annotated[
        Path | None,
        typer.Option(
            "--reference-raster",
            help="Snap the output grid to this georeferenced raster's origin and resolution.",
            show_default=False,
        ),
    ] = None,
    snap_to_grid: Annotated[
        bool,
        typer.Option(
            "--snap-to-grid/--no-snap-to-grid",
            help="Snap the output origin to a whole multiple of the resolution.",
        ),
    ] = True,
    bands: BandsOption = None,
    nodata: Annotated[
        float | None,
        typer.Option(
            "--nodata",
            help="Destination NoData; omit to reuse the source NoData when defined.",
            show_default=False,
        ),
    ] = None,
    product_id: Annotated[
        str | None,
        typer.Option(
            "--product-id",
            help="Filename token; derived from the input path when omitted.",
            show_default=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing reprojected GeoTIFF."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Reproject a map-geometry raster, optionally aligning to a reference grid.

    This is map-to-map warping only. Sensor-geometry products (RPC, no affine CRS) are
    refused — use orthorectification for those. Categorical rasters (quality masks) should
    pass ``--data-semantics categorical`` so class codes are nearest-neighbour resampled.
    """
    request = _build_model(
        ReprojectRequest,
        product_path=input_path,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite),
        target_crs=target_crs,
        resolution=resolution,
        resampling=resampling,
        data_semantics=data_semantics,
        reference_raster=reference_raster,
        snap_to_grid=snap_to_grid,
        bands=_parse_band_indices(bands),
        nodata=nodata,
        product_id=product_id,
        proj_autofix=proj_autofix,
    )
    result = reproject_raster(request)
    payload = {
        "path": str(result.path),
        "crs": result.crs_authority,
        "resolution": result.resolution,
        "width": result.width,
        "height": result.height,
        "band_indices": list(result.band_indices),
        "resampling": result.resampling.value,
        "snapped": result.snapped,
        "reference_raster": (
            None if result.reference_raster is None else str(result.reference_raster)
        ),
        "product_id": result.product_id,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"wrote {result.path} "
        f"({result.crs_authority}, {result.resolution:g} units/px, "
        f"{result.width}x{result.height}, {result.resampling.value})"
    )


@app.command("orthorectify")
def orthorectify_command(
    input_path: InputOption,
    dem_path: Annotated[
        Path,
        typer.Option(
            "--dem",
            help="Digital elevation model covering the scene (required).",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory that receives the orthorectified GeoTIFF.",
            show_default=False,
        ),
    ],
    target_crs: Annotated[
        str,
        typer.Option(
            "--target-crs",
            help="Destination CRS: 'auto' (UTM of scene centre) or an authority code "
            "such as EPSG:32633.",
        ),
    ] = "auto",
    resolution: Annotated[
        float,
        typer.Option(
            "--resolution",
            help="Output ground sample distance in target-CRS units (metres for UTM).",
        ),
    ] = 30.0,
    resampling: Annotated[
        ResamplingMethod,
        typer.Option("--resampling", help="nearest | bilinear | cubic."),
    ] = ResamplingMethod.BILINEAR,
    data_semantics: Annotated[
        DataSemantics,
        typer.Option(
            "--data-semantics",
            help="continuous (default) or categorical. Categorical forces nearest.",
        ),
    ] = DataSemantics.CONTINUOUS,
    snap_to_grid: Annotated[
        bool,
        typer.Option(
            "--snap-to-grid/--no-snap-to-grid",
            help="Snap the output origin to a whole multiple of the resolution.",
        ),
    ] = True,
    bands: BandsOption = None,
    nodata: Annotated[
        float | None,
        typer.Option(
            "--nodata",
            help="Destination NoData; omit to reuse the source NoData when defined.",
            show_default=False,
        ),
    ] = None,
    rpc_dem_missing_value: Annotated[
        float | None,
        typer.Option(
            "--rpc-dem-missing-value",
            help="Elevation used where the DEM has NoData (GDAL RPC_DEM_MISSING_VALUE).",
            show_default=False,
        ),
    ] = None,
    rpc_height_scale: Annotated[
        float | None,
        typer.Option(
            "--rpc-height-scale",
            help="Optional RPC_HEIGHT_SCALE in metres.",
            show_default=False,
        ),
    ] = None,
    rpc_dem_apply_vdatum_shift: Annotated[
        bool,
        typer.Option(
            "--rpc-dem-apply-vdatum-shift/--no-rpc-dem-apply-vdatum-shift",
            help="Forward RPC_DEM_APPLY_VDATUM_SHIFT to GDAL (default off).",
        ),
    ] = False,
    warp_memory_mb: Annotated[
        int,
        typer.Option("--warp-memory-mb", min=1, help="GDAL warp memory limit in MiB."),
    ] = 512,
    product_id: Annotated[
        str | None,
        typer.Option(
            "--product-id",
            help="Filename token; derived from the input path when omitted.",
            show_default=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing orthorectified GeoTIFF."),
    ] = False,
    as_json: JsonFlag = False,
    proj_autofix: ProjAutofixOption = True,
) -> None:
    """Orthorectify sensor-geometry imagery with an RPC model and a DEM.

    Both inputs are mandatory. Missing RPC metadata or a DEM that does not cover the
    scene fails loudly — the command never falls back to plain reprojection. See
    docs/orthorectification.md.
    """
    request = _build_model(
        OrthorectifyRequest,
        product_path=input_path,
        dem_path=dem_path,
        output=_build_model(OutputConfig, directory=output_dir, overwrite=overwrite),
        target_crs=target_crs,
        resolution=resolution,
        resampling=resampling,
        data_semantics=data_semantics,
        snap_to_grid=snap_to_grid,
        bands=_parse_band_indices(bands),
        nodata=nodata,
        rpc_options=_build_model(
            RpcTransformerOptions,
            rpc_height_scale=rpc_height_scale,
            rpc_dem_missing_value=rpc_dem_missing_value,
            rpc_dem_apply_vdatum_shift=rpc_dem_apply_vdatum_shift,
        ),
        warp_memory_mb=warp_memory_mb,
        product_id=product_id,
        proj_autofix=proj_autofix,
    )
    result = orthorectify_raster(request)
    payload = {
        "path": str(result.path),
        "crs": result.crs_authority,
        "resolution": result.resolution,
        "width": result.width,
        "height": result.height,
        "band_indices": list(result.band_indices),
        "resampling": result.resampling.value,
        "dem_path": str(result.dem_path),
        "transformer_options": dict(result.transformer_options),
        "product_id": result.product_id,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"wrote {result.path} "
        f"({result.crs_authority}, {result.resolution:g} units/px, "
        f"{result.width}x{result.height}, {result.resampling.value}, "
        f"dem={result.dem_path.name})"
    )


def main() -> None:
    """Entry point that maps domain errors onto documented exit codes."""
    try:
        app()
    except HyperSatError as error:
        # Logging may not be configured yet if the failure happened during option parsing.
        if not logging.getLogger("hypersat").handlers:
            configure_logging()
        logger.error(str(error), extra={"error_type": type(error).__name__})
        typer.echo(f"error: {error}", err=True)
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    main()
