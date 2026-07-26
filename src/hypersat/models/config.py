"""Configuration models.

Only the configuration that the currently implemented commands actually consume lives
here. Stage options for orthorectification, masking, indices and previews are documented
in ``configs/pipeline.example.yaml`` and become models when their stages are implemented,
so that no model exists without code that reads it.

Validation errors from these models are surfaced by the CLI as
:class:`hypersat.exceptions.ConfigurationError`.
"""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator

from hypersat.models.base import StrictModel

__all__ = [
    "InputConfig",
    "OutputConfig",
    "ProductType",
    "ValidationRequest",
    "ValidationRequirements",
]


class ProductType(StrEnum):
    """How the input should be interpreted."""

    AUTO = "auto"
    """Detect from the path: a directory is treated as a product, a file as a raster."""

    ENMAP_L1B = "enmap_l1b"
    """An EnMAP L1B product directory: sensor geometry, RPC model, per-band wavelengths."""

    GEOTIFF = "geotiff"
    """An opaque multi-band raster; no mission-specific metadata interpretation."""


class InputConfig(StrictModel):
    """The product to process and how much of it to look at."""

    path: Path
    product_type: ProductType = ProductType.AUTO
    band_subset: tuple[int, ...] | None = None
    """1-based band indices to restrict processing to. ``None`` means every band."""
    wavelengths_nm: tuple[float, ...] | None = None
    """Operator-supplied centre wavelengths, used only when metadata has none.

    Prefer fixing the product metadata. When set, the length must match the band count,
    which can only be checked against the opened raster, not here.
    """

    @field_validator("path")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("band_subset")
    @classmethod
    def _validate_band_subset(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("band_subset must not be empty; use null to select every band")
        invalid = sorted({index for index in value if index < 1})
        if invalid:
            raise ValueError(f"band indices are 1-based; got {invalid}")
        duplicates = sorted({index for index in value if value.count(index) > 1})
        if duplicates:
            raise ValueError(f"band_subset contains duplicate indices: {duplicates}")
        # Sorted so that downstream windowed reads are sequential on disk.
        return tuple(sorted(value))

    @field_validator("wavelengths_nm")
    @classmethod
    def _validate_wavelengths(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("wavelengths_nm must not be empty; use null to read metadata")
        if any(not math.isfinite(item) or item <= 0.0 for item in value):
            raise ValueError("wavelengths must be finite and positive, in nanometres")
        if any(later <= earlier for earlier, later in pairwise(value)):
            raise ValueError("wavelengths must be strictly increasing")
        return value


class OutputConfig(StrictModel):
    """Where results are written."""

    directory: Path
    overwrite: bool = False
    """When false, a stage refuses to replace an existing output file."""

    @field_validator("directory")
    @classmethod
    def _expand_directory(cls, value: Path) -> Path:
        return value.expanduser()


class ValidationRequirements(StrictModel):
    """Which pre-flight checks are treated as mandatory."""

    require_georeferencing: bool = True
    """Require either an affine transform with a CRS, or an RPC sensor model."""

    require_rpc: bool = False
    """Require an RPC sensor model. Mandatory for orthorectification."""

    require_wavelengths: bool = False
    """Require a usable centre wavelength for every band."""

    max_uncompressed_gb: Annotated[float, Field(gt=0.0)] | None = 16.0
    """Guard against opening a cube far larger than the machine can handle."""


class ValidationRequest(StrictModel):
    """Everything ``hypersat validate`` needs to know."""

    product: InputConfig
    requirements: ValidationRequirements = ValidationRequirements()
    dem_path: Path | None = None
    """DEM to validate. Required before orthorectification can be attempted."""
    output: OutputConfig | None = None
    treat_warnings_as_errors: bool = False
    proj_autofix: bool = True
    """Fall back to rasterio's bundled PROJ database if the configured one is unusable."""

    @field_validator("dem_path")
    @classmethod
    def _expand_dem_path(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None
