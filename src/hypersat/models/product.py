"""Models describing an inspected raster or satellite product.

These are plain data structures: they are produced by :mod:`hypersat.io.inspect` and
consumed by the CLI, the validation stage and (later) the quality-control report. They
import nothing from rasterio, so they can be constructed in tests without a raster.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from hypersat.models.base import StrictModel
from hypersat.models.environment import EnvironmentInfo

__all__ = [
    "BandInfo",
    "CRSInfo",
    "FileInfo",
    "InputKind",
    "InspectionResult",
    "ProductLayout",
    "RPCInfo",
    "RasterInfo",
    "WavelengthSource",
]


class InputKind(StrEnum):
    """Whether the inspected input was a single file or a product directory."""

    FILE = "file"
    DIRECTORY = "directory"


class WavelengthSource(StrEnum):
    """Where a band's centre wavelength was read from.

    Recorded per band because the trustworthiness differs: a dedicated metadata tag is
    authoritative, whereas a value scraped out of a free-text band description is a
    best-effort interpretation.
    """

    BAND_TAG = "band_tag"
    """A ``wavelength`` (or similar) tag in the band's default metadata domain."""

    IMAGERY_DOMAIN = "imagery_domain"
    """GDAL's ``IMAGERY`` domain ``CENTRAL_WAVELENGTH_UM`` tag, converted to nanometres."""

    DESCRIPTION = "description"
    """Parsed out of the band description text, e.g. ``"band 42 (655.4 nm)"``."""

    CONFIG_OVERRIDE = "config_override"
    """Supplied by the operator through configuration, because metadata was unusable."""


class FileInfo(StrictModel):
    """Filesystem facts about a product or raster on disk."""

    path: Path
    size_bytes: int = Field(ge=0)
    size_human: str
    modified_utc: str | None = None
    sha256: str | None = None
    """Only computed when explicitly requested; hashing a multi-GB cube is expensive."""


class CRSInfo(StrictModel):
    """Coordinate reference system of a raster, as far as it could be determined."""

    is_defined: bool
    authority_code: str | None = None
    """Authority string such as ``EPSG:32633``, when the CRS resolves to one."""
    epsg: int | None = None
    wkt: str | None = None
    is_geographic: bool | None = None
    is_projected: bool | None = None
    linear_units: str | None = None
    """Name of the CRS's linear unit (``metre``, ``US survey foot``, ...)."""
    linear_units_factor: float | None = None
    """Metres per CRS unit, so a resolution can be interpreted without guessing."""


class RPCInfo(StrictModel):
    """Presence and plausibility of an RPC sensor model.

    The 80 polynomial coefficients themselves are intentionally not stored here: they are
    noise in a human-readable report and are available from the raster's ``RPC`` metadata
    domain when the orthorectification stage needs them.

    What "usable" means is shaped by how GDAL handles the ``RPC`` domain (verified against
    GDAL 3.12): a set missing required keys is **dropped in full**, so a partial model
    surfaces as "no RPC at all", and a coefficient list shorter than 20 terms is silently
    **zero-padded**. Counting terms can therefore never detect a truncated product on its
    own, so the checks look for degenerate values instead - an all-zero polynomial or a
    zero normalisation scale would make the coordinate transformation undefined.
    """

    available: bool
    is_usable: bool = False
    """True when the model is structurally complete and free of degenerate values."""
    issues: list[str] = []
    """Human-readable reasons the model is not usable; empty when it is."""
    coefficient_counts: dict[str, int] = {}
    """Term count per coefficient set, e.g. ``{"line_num_coeff": 20, ...}``."""
    line_off: float | None = None
    samp_off: float | None = None
    lat_off: float | None = None
    long_off: float | None = None
    height_off: float | None = None
    line_scale: float | None = None
    samp_scale: float | None = None
    lat_scale: float | None = None
    long_scale: float | None = None
    height_scale: float | None = None
    err_bias: float | None = None
    err_rand: float | None = None


class BandInfo(StrictModel):
    """Per-band metadata of a raster."""

    index: int = Field(ge=1, description="1-based band index, as used by GDAL/rasterio.")
    dtype: str
    nodata: float | None = None
    nodata_is_nan: bool = False
    """NaN NoData is reported through this flag, because JSON cannot represent NaN."""
    description: str | None = None
    units: str | None = None
    wavelength_nm: float | None = None
    wavelength_source: WavelengthSource | None = None
    block_shape: tuple[int, int] | None = None
    scale: float | None = None
    offset: float | None = None
    color_interpretation: str | None = None
    mask_flags: list[str] = []
    metadata_domains: list[str] = []
    tags: dict[str, str] = {}


class RasterInfo(StrictModel):
    """Everything the inspection stage can determine about a raster without reading pixels.

    "Without reading pixels" is deliberate: inspection of a multi-gigabyte hyperspectral
    cube must stay fast and memory-safe, so no statistics are computed here. Per-band
    statistics arrive with the analytics milestone.
    """

    path: Path
    driver: str
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    band_count: int = Field(ge=0)
    dtypes: list[str] = []
    nodata: float | None = None
    nodata_is_nan: bool = False
    nodata_per_band: list[float | None] = []
    crs: CRSInfo
    transform: list[float] = Field(
        default=[],
        description="Six affine coefficients in rasterio order: a, b, c, d, e, f.",
    )
    transform_is_identity: bool = True
    has_affine_georeferencing: bool = False
    """True when the raster carries both a CRS and a non-identity affine transform.

    An EnMAP L1B product is expected to be False here while still being georeferenced
    through its RPC sensor model - see ``docs/orthorectification.md``.
    """
    bounds: tuple[float, float, float, float] | None = None
    """``(left, bottom, right, top)`` in CRS units, or ``None`` without georeferencing."""
    pixel_size: tuple[float, float] | None = None
    """``(x_size, y_size)`` in CRS units; both positive."""
    is_tiled: bool = False
    block_shapes: list[tuple[int, int]] = []
    compression: str | None = None
    interleaving: str | None = None
    metadata_domains: list[str] = []
    """Dataset-level metadata domains, e.g. ``["IMAGE_STRUCTURE", "RPC"]``."""
    metadata: dict[str, str] = {}
    """Tags of the default metadata domain."""
    rpc: RPCInfo
    gcp_count: int = 0
    bands: list[BandInfo] = []
    file: FileInfo
    sidecar_files: list[Path] = []
    """Additional files the driver depends on (ENVI header, world file, ...)."""
    estimated_uncompressed_bytes: int = Field(ge=0)
    """``width * height * band_count * itemsize``: the cost of a naive full read."""
    warnings: list[str] = []

    @property
    def wavelengths_nm(self) -> list[float] | None:
        """Centre wavelengths of all bands, or ``None`` if any band lacks one."""
        values = [band.wavelength_nm for band in self.bands]
        if not values or any(value is None for value in values):
            return None
        return [value for value in values if value is not None]


class ProductLayout(StrictModel):
    """Files discovered inside a satellite-product directory.

    Only enough structure to locate the raster and report what is present. Parsing
    EnMAP's metadata XML is a later milestone; nothing here claims to understand the
    mission's product specification.
    """

    root: Path
    raster_candidates: list[Path] = []
    metadata_files: list[Path] = []
    other_files_count: int = 0
    total_size_bytes: int = Field(ge=0)
    total_size_human: str


class InspectionResult(StrictModel):
    """Complete result of ``hypersat inspect``."""

    input_path: Path
    input_kind: InputKind
    resolved_raster_path: Path
    raster: RasterInfo
    product: ProductLayout | None = None
    environment: EnvironmentInfo
    warnings: list[str] = []
