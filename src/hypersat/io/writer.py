"""Atomic GeoTIFF writing that carries metadata through unchanged.

Two properties matter here and both are about not lying to whoever reads the output later.

**Atomicity.** A raster is written to a temporary sibling and moved into place with
``Path.replace`` only after the dataset has closed cleanly. A crash, a full disk or a
cancelled run therefore leaves either the previous file or nothing at all - never a
half-written GeoTIFF that opens fine, looks plausible and is missing its last blocks. The
temporary file is a sibling rather than a file in the system temp directory because
``replace`` is only atomic within one filesystem.

**Metadata propagation.** CRS, affine transform, NoData, band descriptions and centre
wavelengths are written back out, so a windowed subset stays as interpretable as its
source. The one thing deliberately *not* propagated by default is the RPC sensor model:
its coefficients are expressed against the source line/sample origin, so attaching them to
a subset would describe geometry the pixels do not have. :func:`write_chunk` drops them,
with a warning, for anything that is not a full-grid read.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.crs import CRS
from rasterio.errors import CRSError, NotGeoreferencedWarning, RasterioIOError
from rasterio.transform import Affine

from hypersat.exceptions import RasterWriteError
from hypersat.logging_config import get_logger
from hypersat.models.raster import RasterChunk, RasterMetadata

__all__ = [
    "BLOCK_SIZE_MULTIPLE",
    "DEFAULT_BLOCK_SIZE",
    "Compression",
    "GeoTiffOptions",
    "write_array",
    "write_chunk",
]

logger = get_logger(__name__)

DEFAULT_BLOCK_SIZE = 256
"""Tile edge in pixels. 256 keeps a single tile small while limiting directory overhead."""

BLOCK_SIZE_MULTIPLE = 16
"""The TIFF specification requires tile dimensions to be a multiple of 16."""

_VALID_BIGTIFF = frozenset({"YES", "NO", "IF_NEEDED", "IF_SAFER"})
_PREDICTOR_NONE = 1
_PREDICTOR_HORIZONTAL = 2
_PREDICTOR_FLOATING_POINT = 3
_PREDICTOR_COMPRESSIONS = frozenset({"DEFLATE", "LZW", "ZSTD"})
_NDIM_SINGLE_BAND = 2
_NDIM_BAND_CUBE = 3


class Compression(StrEnum):
    """GeoTIFF compression algorithms this project writes.

    DEFLATE is the default because every GDAL build and every desktop GIS reads it. ZSTD
    is faster at a similar ratio and is available in the rasterio wheels, but it needs a
    reasonably recent GDAL on the reading side, which an output product cannot assume.
    """

    NONE = "NONE"
    DEFLATE = "DEFLATE"
    LZW = "LZW"
    ZSTD = "ZSTD"


@dataclass(frozen=True, slots=True)
class GeoTiffOptions:
    """Storage-layout choices for an output GeoTIFF.

    These affect only how bytes are arranged on disk; no option here changes a sample
    value.

    Attributes:
        compression: Compression algorithm.
        tiled: Write tiles rather than strips. Tiles make windowed reads cheap, which is
            what every later stage does.
        block_size: Tile edge in pixels; must be a positive multiple of 16.
        bigtiff: GDAL's ``BIGTIFF`` creation option. ``IF_SAFER`` switches to the 64-bit
            layout whenever the result might approach the 4 GB classic-TIFF limit, which a
            hyperspectral cube does.
        predictor: TIFF predictor, or ``None`` to choose from the dtype: horizontal
            differencing for integers, floating-point predictor for floats. Both are
            lossless and typically shave a good fraction off the compressed size.
    """

    compression: Compression = Compression.DEFLATE
    tiled: bool = True
    block_size: int = DEFAULT_BLOCK_SIZE
    bigtiff: str = "IF_SAFER"
    predictor: int | None = None

    def __post_init__(self) -> None:
        """Validate options that GDAL would otherwise reject late or silently ignore.

        Raises:
            ValueError: If the block size or the BIGTIFF mode is invalid.
        """
        if self.block_size <= 0 or self.block_size % BLOCK_SIZE_MULTIPLE:
            raise ValueError(
                f"block_size must be a positive multiple of {BLOCK_SIZE_MULTIPLE}, "
                f"got {self.block_size}"
            )
        if self.bigtiff.upper() not in _VALID_BIGTIFF:
            raise ValueError(f"bigtiff must be one of {sorted(_VALID_BIGTIFF)}, got {self.bigtiff}")

    def predictor_for(self, dtype: np.dtype[Any]) -> int:
        """Return the predictor to use for a dtype.

        Args:
            dtype: dtype of the array being written.

        Returns:
            The explicit predictor when one was configured, otherwise 3 for floating-point
            data, 2 for integers, and 1 when the compression does not support prediction.
        """
        if self.compression.value not in _PREDICTOR_COMPRESSIONS:
            return _PREDICTOR_NONE
        if self.predictor is not None:
            return self.predictor
        if np.issubdtype(dtype, np.floating):
            return _PREDICTOR_FLOATING_POINT
        return _PREDICTOR_HORIZONTAL


def _as_band_cube(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return ``data`` shaped ``(bands, rows, columns)``.

    Raises:
        RasterWriteError: If the array is neither two- nor three-dimensional.
    """
    if data.ndim == _NDIM_SINGLE_BAND:
        return data[np.newaxis, :, :]
    if data.ndim == _NDIM_BAND_CUBE:
        return data
    raise RasterWriteError(
        "Raster data must be 2- or 3-dimensional.",
        hint="Pass a (rows, columns) array for one band or (bands, rows, columns) for several.",
        context={"ndim": int(data.ndim), "shape": [int(size) for size in data.shape]},
    )


def _materialise(data: npt.NDArray[Any], nodata: float | None) -> npt.NDArray[Any]:
    """Turn a possibly-masked array into the plain array GDAL writes.

    Raises:
        RasterWriteError: If masked samples cannot be represented in the output.
    """
    if not np.ma.isMaskedArray(data):
        return data
    if nodata is None:
        raise RasterWriteError(
            "Masked data cannot be written without a NoData value.",
            hint="Set metadata.nodata so masked samples have a representation in the "
            "file, or fill the mask yourself before writing.",
            context={"dtype": str(data.dtype)},
        )
    filled: npt.NDArray[Any] = np.ma.filled(data, nodata)
    return filled


def _build_profile(
    data: npt.NDArray[Any],
    metadata: RasterMetadata,
    options: GeoTiffOptions,
) -> dict[str, Any]:
    """Assemble the rasterio creation profile for the output file.

    Raises:
        RasterWriteError: If the CRS or transform cannot be interpreted.
    """
    count, height, width = data.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": str(data.dtype),
        "tiled": options.tiled,
        "BIGTIFF": options.bigtiff.upper(),
    }
    if options.tiled:
        profile["blockxsize"] = options.block_size
        profile["blockysize"] = options.block_size
    if options.compression is not Compression.NONE:
        profile["compress"] = options.compression.value
        predictor = options.predictor_for(data.dtype)
        if predictor != _PREDICTOR_NONE:
            profile["predictor"] = predictor
    if metadata.nodata is not None:
        profile["nodata"] = metadata.nodata
    if metadata.crs_wkt is not None:
        try:
            profile["crs"] = CRS.from_wkt(metadata.crs_wkt)
        except CRSError as error:
            raise RasterWriteError(
                "The coordinate reference system could not be interpreted.",
                hint="metadata.crs_wkt must be valid WKT; a stale or partial PROJ "
                "database also causes this (see the README troubleshooting section).",
                context={"reason": str(error)},
            ) from error
    if metadata.transform is not None:
        profile["transform"] = Affine(*metadata.transform)
    return profile


def _apply_band_metadata(dataset: Any, metadata: RasterMetadata, count: int) -> None:
    """Write per-band descriptions and wavelength tags onto an open dataset."""
    for position in range(count):
        band_index = position + 1
        if position < len(metadata.band_descriptions):
            description = metadata.band_descriptions[position]
            if description:
                dataset.set_band_description(band_index, description)
        if position < len(metadata.wavelengths_nm):
            wavelength = metadata.wavelengths_nm[position]
            if wavelength is not None:
                dataset.update_tags(
                    band_index,
                    wavelength=f"{wavelength:g}",
                    wavelength_units="nm",
                )


def _write_dataset(
    destination: Path,
    data: npt.NDArray[Any],
    metadata: RasterMetadata,
    options: GeoTiffOptions,
) -> None:
    """Create the output file and write pixels and metadata into it.

    Raises:
        RasterWriteError: If the dataset cannot be created or written.
    """
    profile = _build_profile(data, metadata, options)
    try:
        # A product in sensor geometry legitimately has no transform, and rasterio warns
        # on every such creation. The condition is already visible in the output's own
        # metadata, so the warning is noise here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(destination, "w", **profile) as dataset:
                dataset.write(data)
                _apply_band_metadata(dataset, metadata, int(data.shape[0]))
                if metadata.dataset_tags:
                    dataset.update_tags(**dict(metadata.dataset_tags))
                if metadata.rpc_tags:
                    dataset.update_tags(ns="RPC", **dict(metadata.rpc_tags))
    except (RasterioIOError, ValueError, OSError) as error:
        raise RasterWriteError(
            "Could not write the raster.",
            hint="Check free space and write permission on the destination directory.",
            context={"path": str(destination), "reason": str(error)},
        ) from error


def write_array(
    path: Path,
    data: npt.NDArray[Any],
    *,
    metadata: RasterMetadata | None = None,
    options: GeoTiffOptions | None = None,
    overwrite: bool = False,
) -> Path:
    """Write an array to a GeoTIFF atomically.

    Args:
        path: Destination file. Its parent directory must already exist.
        data: ``(rows, columns)`` or ``(bands, rows, columns)`` array. A masked array is
            filled with ``metadata.nodata``.
        metadata: Georeferencing and per-band metadata to attach; defaults to none, which
            produces a plain non-georeferenced raster.
        options: Storage layout; defaults to tiled DEFLATE with a dtype-appropriate
            predictor.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        The path written.

    Raises:
        RasterWriteError: If the destination is unusable, the data or metadata cannot be
            written, or the file exists and ``overwrite`` is false.
    """
    effective_metadata = metadata if metadata is not None else RasterMetadata()
    effective_options = options if options is not None else GeoTiffOptions()

    if path.exists() and not overwrite:
        raise RasterWriteError(
            "The output file already exists.",
            hint="Pass overwrite=True, or choose a different destination.",
            context={"path": str(path)},
        )
    if not path.parent.is_dir():
        raise RasterWriteError(
            "The output directory does not exist.",
            hint="Create it first, or run `hypersat validate --output-dir <dir>`, which "
            "checks that the directory exists and is writable.",
            context={"path": str(path), "directory": str(path.parent)},
        )

    cube = _as_band_cube(data)
    materialised = _materialise(cube, effective_metadata.nodata)

    # Same directory, so the final move stays on one filesystem and is therefore atomic.
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{id(data):x}")
    try:
        _write_dataset(temporary, materialised, effective_metadata, effective_options)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    logger.info(
        "wrote raster",
        extra={
            "path": str(path),
            "bands": int(materialised.shape[0]),
            "width": int(materialised.shape[2]),
            "height": int(materialised.shape[1]),
            "dtype": str(materialised.dtype),
            "compression": effective_options.compression.value,
        },
    )
    return path


def write_chunk(
    path: Path,
    chunk: RasterChunk,
    *,
    options: GeoTiffOptions | None = None,
    overwrite: bool = False,
    keep_rpc: bool = True,
) -> Path:
    """Write a chunk to a GeoTIFF, propagating its metadata.

    The RPC sensor model travels with the pixels only when the chunk covers the whole
    source grid. For a windowed subset the coefficients would point at the wrong line and
    sample origin, and a file that carries a *wrong* sensor model is worse than one that
    carries none, because orthorectification would happily run on it.

    Args:
        path: Destination file.
        chunk: Pixels and metadata to write.
        options: Storage layout; defaults to tiled DEFLATE.
        overwrite: Whether an existing destination may be replaced.
        keep_rpc: Whether to carry the sensor model over when the grid allows it.

    Returns:
        The path written.

    Raises:
        RasterWriteError: If the raster cannot be written.
    """
    metadata = chunk.metadata
    if metadata.rpc_tags is not None:
        drop_reason: str | None = None
        if not keep_rpc:
            drop_reason = "keep_rpc=False"
        elif not chunk.covers_full_grid:
            drop_reason = "the chunk is a windowed subset of the source grid"
        if drop_reason is not None:
            logger.warning(
                "dropping the RPC sensor model from the output",
                extra={"path": str(path), "reason": drop_reason},
            )
            metadata = _without_rpc(metadata)

    return write_array(
        path,
        chunk.data,
        metadata=metadata,
        options=options,
        overwrite=overwrite,
    )


def _without_rpc(metadata: RasterMetadata) -> RasterMetadata:
    """Return a copy of ``metadata`` with the sensor model removed."""
    return RasterMetadata(
        crs_wkt=metadata.crs_wkt,
        transform=metadata.transform,
        nodata=metadata.nodata,
        band_descriptions=metadata.band_descriptions,
        wavelengths_nm=metadata.wavelengths_nm,
        dataset_tags=metadata.dataset_tags,
        rpc_tags=None,
    )
