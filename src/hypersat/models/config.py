"""Configuration models.

Only the configuration that the currently implemented commands actually consume lives
here. Stage options for orthorectification and the YAML pipeline runner remain documented
in ``configs/pipeline.example.yaml`` until those stages are implemented, so that no model
exists without code that reads it.

Validation errors from these models are surfaced by the CLI as
:class:`hypersat.exceptions.ConfigurationError`.
"""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from hypersat.models.base import StrictModel

__all__ = [
    "DataSemantics",
    "IndexRequest",
    "InputConfig",
    "MorphologyConfig",
    "MorphologyKernelShape",
    "MorphologyOperation",
    "OutputConfig",
    "PreviewComposite",
    "PreviewRequest",
    "ProductType",
    "QualityMaskRequest",
    "ReprojectRequest",
    "ResamplingMethod",
    "SpectralIndexName",
    "SpectralProfileRequest",
    "StretchConfig",
    "ValidationRequest",
    "ValidationRequirements",
]

DEFAULT_LOWER_PERCENTILE = 2.0
DEFAULT_UPPER_PERCENTILE = 98.0
RGB_BAND_COUNT = 3
DEFAULT_INDEX_TOLERANCE_NM = 15.0
DEFAULT_RED_NM = 665.0
DEFAULT_NIR_NM = 842.0
DEFAULT_GREEN_NM = 560.0
DEFAULT_INDEX_NODATA = -9999.0


class ProductType(StrEnum):
    """How the input should be interpreted."""

    AUTO = "auto"
    """Detect from the path: a directory is treated as a product, a file as a raster."""

    ENMAP_L1B = "enmap_l1b"
    """An EnMAP L1B product directory: sensor geometry, RPC model, per-band wavelengths."""

    GEOTIFF = "geotiff"
    """An opaque multi-band raster; no mission-specific metadata interpretation."""


class PreviewComposite(StrEnum):
    """Which cosmetic composite a preview request produces."""

    TRUE_COLOR = "true-color"
    FALSE_COLOR = "false-color"
    BAND = "band"


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


class StretchConfig(StrictModel):
    """Percentile stretch applied when building a cosmetic preview."""

    lower_percentile: Annotated[float, Field(ge=0.0, lt=100.0)] = DEFAULT_LOWER_PERCENTILE
    upper_percentile: Annotated[float, Field(gt=0.0, le=100.0)] = DEFAULT_UPPER_PERCENTILE
    per_band: bool = True
    """When true, each band is stretched independently; when false, one shared range is used."""

    @model_validator(mode="after")
    def _ordered_percentiles(self) -> Self:
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be strictly less than upper_percentile")
        return self


class PreviewRequest(StrictModel):
    """Everything ``hypersat preview`` needs to know.

    Band order is significant for RGB composites, so this model keeps an ordered
    ``bands`` field instead of reusing :attr:`InputConfig.band_subset`, which sorts.
    """

    product_path: Path
    output: OutputConfig
    composite: PreviewComposite = PreviewComposite.TRUE_COLOR
    bands: tuple[int, ...] | None = None
    """Explicit 1-based band indices in display order. Overrides wavelength selection."""
    band: Annotated[int, Field(ge=1)] | None = None
    """Explicit band for :attr:`PreviewComposite.BAND` when ``bands`` is omitted."""
    stretch: StretchConfig = StretchConfig()
    max_dimension: Annotated[int, Field(gt=0)] = 2048
    blur_kernel: Annotated[int, Field(gt=0)] | None = None
    product_id: str | None = None
    proj_autofix: bool = True
    wavelength_tolerance_nm: Annotated[float, Field(ge=0.0)] = 30.0

    @field_validator("product_path")
    @classmethod
    def _expand_product_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("bands")
    @classmethod
    def _validate_bands(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("bands must not be empty")
        invalid = sorted({index for index in value if index < 1})
        if invalid:
            raise ValueError(f"band indices are 1-based; got {invalid}")
        return value

    @field_validator("blur_kernel")
    @classmethod
    def _odd_blur_kernel(cls, value: int | None) -> int | None:
        if value is not None and value % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        return value

    @field_validator("product_id")
    @classmethod
    def _validate_product_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product_id must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _composite_arguments(self) -> Self:
        if self.composite is PreviewComposite.BAND:
            if self.bands is None and self.band is None:
                raise ValueError("composite 'band' requires band or bands")
            if self.bands is not None and len(self.bands) != 1:
                raise ValueError("composite 'band' expects exactly one band index")
        elif self.bands is not None and len(self.bands) != RGB_BAND_COUNT:
            raise ValueError("RGB composites expect exactly three band indices")
        return self


class SpectralIndexName(StrEnum):
    """Supported conventional spectral indices."""

    NDVI = "ndvi"
    NDWI = "ndwi"


class IndexRequest(StrictModel):
    """Everything ``hypersat calculate-index`` needs to know."""

    product_path: Path
    output: OutputConfig
    index: SpectralIndexName
    red_nm: Annotated[float, Field(gt=0.0)] = DEFAULT_RED_NM
    nir_nm: Annotated[float, Field(gt=0.0)] = DEFAULT_NIR_NM
    green_nm: Annotated[float, Field(gt=0.0)] = DEFAULT_GREEN_NM
    tolerance_nm: Annotated[float, Field(ge=0.0)] = DEFAULT_INDEX_TOLERANCE_NM
    output_nodata: float = DEFAULT_INDEX_NODATA
    bands: tuple[int, int] | None = None
    """Optional explicit 1-based band pair ``(minuend, subtrahend)`` overriding wavelengths."""
    product_id: str | None = None
    include_statistics: bool = False
    statistics_sample_step: Annotated[int, Field(ge=1)] = 1
    proj_autofix: bool = True

    @field_validator("product_path")
    @classmethod
    def _expand_product_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("bands")
    @classmethod
    def _validate_band_pair(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if any(index < 1 for index in value):
            raise ValueError("band indices are 1-based")
        return value

    @field_validator("product_id")
    @classmethod
    def _validate_product_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product_id must not be empty")
        return cleaned


class SpectralProfileRequest(StrictModel):
    """Everything ``hypersat spectral-profile`` needs to know."""

    product_path: Path
    output: OutputConfig
    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]
    window_size: Annotated[int, Field(ge=1)] = 1
    bands: tuple[int, ...] | None = None
    product_id: str | None = None
    proj_autofix: bool = True

    @field_validator("product_path")
    @classmethod
    def _expand_product_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("window_size")
    @classmethod
    def _odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        return value

    @field_validator("bands")
    @classmethod
    def _validate_bands(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("bands must not be empty")
        invalid = sorted({index for index in value if index < 1})
        if invalid:
            raise ValueError(f"band indices are 1-based; got {invalid}")
        return value

    @field_validator("product_id")
    @classmethod
    def _validate_product_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product_id must not be empty")
        return cleaned


class MorphologyOperation(StrEnum):
    """OpenCV morphology operations available for the quality mask."""

    NONE = "none"
    OPEN = "open"
    CLOSE = "close"
    DILATE = "dilate"
    ERODE = "erode"


class MorphologyKernelShape(StrEnum):
    """Structuring-element shapes accepted by OpenCV."""

    RECT = "rect"
    ELLIPSE = "ellipse"
    CROSS = "cross"


class MorphologyConfig(StrictModel):
    """Optional morphological post-processing of defect classes.

    Disabled by default. When enabled, every parameter is explicit so the QC report can
    state exactly what was done.
    """

    enabled: bool = False
    operation: MorphologyOperation = MorphologyOperation.CLOSE
    kernel_shape: MorphologyKernelShape = MorphologyKernelShape.ELLIPSE
    kernel_size: Annotated[int, Field(ge=1)] = 3
    iterations: Annotated[int, Field(ge=1)] = 1

    @field_validator("kernel_size")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return value


class QualityMaskRequest(StrictModel):
    """Everything ``hypersat quality-mask`` needs to know."""

    product_path: Path
    output: OutputConfig
    saturation_dn: float = 65535.0
    low_signal_dn: float = 10.0
    evaluation_wavelengths_nm: tuple[float, ...] | None = (490.0, 560.0, 665.0, 842.0)
    saturation_band_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    bands: tuple[int, ...] | None = None
    """Optional explicit evaluation bands; overrides wavelength selection."""
    tolerance_nm: Annotated[float, Field(ge=0.0)] = 20.0
    morphology: MorphologyConfig = MorphologyConfig()
    spectral_anomaly: bool = False
    anomaly_cv_threshold: Annotated[float, Field(gt=0.0)] = 2.0
    product_id: str | None = None
    include_statistics: bool = False
    proj_autofix: bool = True

    @field_validator("product_path")
    @classmethod
    def _expand_product_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("evaluation_wavelengths_nm")
    @classmethod
    def _validate_wavelengths(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("evaluation_wavelengths_nm must not be empty; use null for every band")
        if any(not math.isfinite(item) or item <= 0.0 for item in value):
            raise ValueError("evaluation wavelengths must be finite and positive")
        return value

    @field_validator("bands")
    @classmethod
    def _validate_bands(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("bands must not be empty")
        invalid = sorted({index for index in value if index < 1})
        if invalid:
            raise ValueError(f"band indices are 1-based; got {invalid}")
        return value

    @field_validator("product_id")
    @classmethod
    def _validate_product_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product_id must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> Self:
        if self.low_signal_dn >= self.saturation_dn:
            raise ValueError("low_signal_dn must be strictly less than saturation_dn")
        return self


class ResamplingMethod(StrEnum):
    """Supported warping kernels for continuous imagery."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"


class DataSemantics(StrEnum):
    """How sample values should be treated during resampling.

    Categorical data (quality masks, class maps) must use nearest-neighbour so that
    class codes are never averaged into meaningless intermediates.
    """

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class ReprojectRequest(StrictModel):
    """Everything ``hypersat reproject`` needs to know.

    Either ``target_crs`` (including ``auto`` for UTM) or ``reference_raster`` defines
    the destination grid. When both are set, the reference supplies the grid and
    ``target_crs`` must match the reference CRS (``auto`` means "use the reference").
    """

    product_path: Path
    output: OutputConfig
    target_crs: str = "auto"
    resolution: Annotated[float, Field(gt=0.0)] | None = None
    resampling: ResamplingMethod = ResamplingMethod.BILINEAR
    data_semantics: DataSemantics = DataSemantics.CONTINUOUS
    reference_raster: Path | None = None
    snap_to_grid: bool = True
    bands: tuple[int, ...] | None = None
    nodata: float | None = None
    product_id: str | None = None
    proj_autofix: bool = True

    @field_validator("product_path")
    @classmethod
    def _expand_product_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("reference_raster")
    @classmethod
    def _expand_reference(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser()

    @field_validator("target_crs")
    @classmethod
    def _validate_target_crs(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("target_crs must not be empty; use 'auto' or an authority code")
        return cleaned

    @field_validator("bands")
    @classmethod
    def _validate_bands(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("bands must not be empty")
        invalid = sorted({index for index in value if index < 1})
        if invalid:
            raise ValueError(f"band indices are 1-based; got {invalid}")
        return value

    @field_validator("product_id")
    @classmethod
    def _validate_product_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product_id must not be empty")
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def _categorical_forces_nearest(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        semantics = data.get("data_semantics", DataSemantics.CONTINUOUS)
        if semantics in (DataSemantics.CATEGORICAL, DataSemantics.CATEGORICAL.value):
            data = {**data, "resampling": ResamplingMethod.NEAREST}
        return data
