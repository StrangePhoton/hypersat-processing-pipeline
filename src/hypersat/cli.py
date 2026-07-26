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
    InputConfig,
    OutputConfig,
    ValidationRequest,
    ValidationRequirements,
)
from hypersat.processing.validation import raise_if_invalid, validate_request

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
