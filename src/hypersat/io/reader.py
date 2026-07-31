"""Windowed, band-selective raster reading with an explicit memory budget.

A hyperspectral cube is the reason this module exists. An EnMAP L1B scene is roughly
1000 x 1024 pixels across 224 bands of ``uint16``: about 460 MB as stored and 920 MB once
converted to ``float32``, before any intermediate array an algorithm allocates. Reading
such a file with a bare ``dataset.read()`` is how a pipeline dies on a machine that was
sized for the job, so every read here goes through a budget check that fails loudly and
suggests a window instead of allocating first and hoping.

Three deliberate behaviours:

* **Masked by default.** NoData is metadata, not a pixel value. A masked array keeps that
  distinction, whereas filling with NaN produces a value that spreads silently through
  arithmetic. NaN filling is available, but only when asked for.
* **Band order is preserved.** ``bands=[42, 30, 20]`` returns them in that order, because
  the caller is usually building a composite where order is the point. This differs from
  :func:`hypersat.io.inspect.inspect_raster`, which sorts, since a report has no order
  semantics.
* **Reads follow the file's own block grid.** GDAL decompresses whole blocks, so
  :func:`iter_block_windows` hands back the tiling the file actually uses rather than an
  arbitrary chunking that would re-read the same bytes repeatedly.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.errors import CRSError, NotGeoreferencedWarning, RasterioIOError
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from hypersat.exceptions import MemoryBudgetExceededError, RasterReadError
from hypersat.io.files import format_bytes
from hypersat.io.inspect import extract_wavelength_nm
from hypersat.logging_config import get_logger
from hypersat.models.raster import RasterChunk, RasterMetadata, ReadWindow

__all__ = [
    "DEFAULT_READ_BUDGET_BYTES",
    "MASK_BYTES_PER_PIXEL",
    "ReadOptions",
    "estimate_read_bytes",
    "iter_block_windows",
    "iter_chunks",
    "open_raster",
    "read_chunk",
]

logger = get_logger(__name__)

DEFAULT_READ_BUDGET_BYTES = 1 << 30
"""One gibibyte: comfortably larger than any sensible window, far smaller than a full cube.

Chosen so that the guard never interferes with normal work but still stops a whole-file
read of a real hyperspectral product before it allocates.
"""

MASK_BYTES_PER_PIXEL = 1
"""A masked read also allocates a boolean mask, one byte per sample, alongside the data."""


@dataclass(frozen=True, slots=True)
class ReadOptions:
    """How pixels should be materialised, independent of *which* pixels are read.

    Attributes:
        masked: Return a ``numpy.ma.MaskedArray`` with NoData masked out.
        as_float32: Convert samples to ``float32`` while reading. GDAL performs the
            conversion, so no intermediate array of the source dtype is allocated.
        fill_nan: Replace masked samples with NaN and return a plain array. Implies
            ``as_float32``, because NaN needs a floating-point dtype, and requires
            ``masked`` so that there is a mask to apply.
        max_bytes: Refuse reads whose result would exceed this many bytes. ``None``
            disables the guard, which is only appropriate when the caller has already
            reasoned about the size.
    """

    masked: bool = True
    as_float32: bool = False
    fill_nan: bool = False
    max_bytes: int | None = DEFAULT_READ_BUDGET_BYTES

    def __post_init__(self) -> None:
        """Reject combinations that cannot be honoured.

        Raises:
            ValueError: If NaN filling is requested without a masked read, or the budget
                is not positive.
        """
        if self.fill_nan and not self.masked:
            raise ValueError("fill_nan requires masked=True; there is no mask to apply otherwise")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive or None, got {self.max_bytes}")

    @property
    def output_dtype(self) -> str | None:
        """Return the dtype to read into, or ``None`` to keep the source dtype."""
        return "float32" if (self.as_float32 or self.fill_nan) else None


@contextmanager
def open_raster(path: Path) -> Iterator[Any]:
    """Open a raster for reading, translating rasterio failures into domain errors.

    Imagery in sensor geometry has no affine transform and rasterio warns on every open.
    That is the expected shape of an L1B product rather than a problem, and
    ``hypersat inspect`` already reports it as a field, so the warning is suppressed here
    instead of being printed once per window.

    Args:
        path: Raster file to open.

    Yields:
        The open ``rasterio`` dataset.

    Raises:
        RasterReadError: If the file cannot be opened.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            dataset = rasterio.open(path)
    except RasterioIOError as error:
        raise RasterReadError(
            "Could not open the raster for reading.",
            hint="Check the path exists and is a format GDAL can read; run "
            "`hypersat inspect --input <path>` to see what the driver reports.",
            context={"path": str(path), "reason": str(error)},
        ) from error
    try:
        yield dataset
    finally:
        dataset.close()


def estimate_read_bytes(
    window: ReadWindow,
    band_count: int,
    dtype: str,
    *,
    masked: bool = False,
) -> int:
    """Return the memory a read would need, before any of it is allocated.

    Args:
        window: Region to be read.
        band_count: Number of bands to be read.
        dtype: NumPy dtype name the samples will be materialised as.
        masked: Whether a boolean mask will be allocated alongside the samples.

    Returns:
        Size in bytes of the resulting array, including the mask when applicable.
    """
    itemsize = int(np.dtype(dtype).itemsize)
    if masked:
        itemsize += MASK_BYTES_PER_PIXEL
    return window.pixel_count * band_count * itemsize


def _check_budget(estimate: int, options: ReadOptions, path: Path, window: ReadWindow) -> None:
    """Refuse a read that would exceed the configured budget.

    Raises:
        MemoryBudgetExceededError: If ``estimate`` exceeds ``options.max_bytes``.
    """
    if options.max_bytes is None or estimate <= options.max_bytes:
        return
    raise MemoryBudgetExceededError(
        f"Reading this region would need {format_bytes(estimate)}, which exceeds the "
        f"{format_bytes(options.max_bytes)} budget.",
        hint="Read a smaller window, select fewer bands, iterate with `iter_chunks`, or "
        "raise ReadOptions.max_bytes if this much memory really is available.",
        context={
            "path": str(path),
            "estimate_bytes": estimate,
            "budget_bytes": options.max_bytes,
            "window": [window.col_off, window.row_off, window.width, window.height],
        },
    )


def _validated_band_indices(band_count: int, bands: Sequence[int] | None) -> tuple[int, ...]:
    """Return the one-based band indices to read, preserving the caller's order.

    Raises:
        RasterReadError: If the selection is empty or refers to bands the raster lacks.
    """
    if bands is None:
        return tuple(range(1, band_count + 1))
    if not bands:
        raise RasterReadError(
            "No bands were requested.",
            hint="Pass at least one 1-based band index, or None to read every band.",
            context={"band_count": band_count},
        )
    out_of_range = sorted({index for index in bands if index < 1 or index > band_count})
    if out_of_range:
        raise RasterReadError(
            "Requested band indices are outside the raster's band range.",
            hint=f"This raster has {band_count} band(s); indices are 1-based.",
            context={"requested": list(bands), "out_of_range": out_of_range},
        )
    return tuple(bands)


def _validated_window(dataset: Any, window: ReadWindow | None, path: Path) -> ReadWindow:
    """Return the window to read, defaulting to the whole raster.

    Raises:
        RasterReadError: If the window extends beyond the raster.
    """
    width = int(dataset.width)
    height = int(dataset.height)
    if window is None:
        return ReadWindow.covering(width, height)
    if not window.fits_within(width, height):
        raise RasterReadError(
            "The requested window extends beyond the raster.",
            hint="Clip the window to the raster size, or use `iter_block_windows` to "
            "walk the file's own block grid.",
            context={
                "path": str(path),
                "window": [window.col_off, window.row_off, window.width, window.height],
                "raster_size": [width, height],
            },
        )
    return window


def _widest_dtype(dataset: Any, indices: Sequence[int]) -> str:
    """Return the largest dtype among the selected bands, which sizes the read."""
    dtypes = [str(dataset.dtypes[index - 1]) for index in indices]
    return max(dtypes, key=lambda name: np.dtype(name).itemsize)


def _nodata_for(dataset: Any, indices: Sequence[int], path: Path) -> float | None:
    """Return the NoData value shared by the selected bands, warning when they disagree."""
    values = {dataset.nodatavals[index - 1] for index in indices}
    distinct = {None if value is None else float(value) for value in values}
    if len(distinct) > 1:
        logger.warning(
            "selected bands declare different NoData values; using the first band's",
            extra={
                "path": str(path),
                "bands": list(indices),
                "nodata_values": sorted(str(value) for value in distinct),
            },
        )
    first = dataset.nodatavals[indices[0] - 1]
    return None if first is None else float(first)


def _crs_wkt(dataset: Any, path: Path) -> str | None:
    """Return the dataset CRS as WKT, or ``None`` when it is absent or unreadable."""
    try:
        crs = dataset.crs
    except CRSError as error:  # pragma: no cover - needs a broken PROJ database
        logger.warning("CRS could not be read", extra={"path": str(path), "reason": str(error)})
        return None
    return None if crs is None else str(crs.to_wkt())


def _chunk_metadata(
    dataset: Any, indices: Sequence[int], window: Window, path: Path
) -> RasterMetadata:
    """Collect the georeferencing and per-band metadata that travels with the pixels."""
    source_transform = dataset.transform
    transform: tuple[float, ...] | None = None
    if not source_transform.is_identity:
        # Shifting the origin to the window is what keeps a subset georeferenced. Doing it
        # to an identity transform would instead invent a translation for imagery that has
        # no map geometry at all, so sensor-geometry input keeps `None`.
        transform = tuple(
            float(value) for value in tuple(window_transform(window, source_transform))[:6]
        )

    descriptions: list[str | None] = []
    wavelengths: list[float | None] = []
    for index in indices:
        description = dataset.descriptions[index - 1]
        wavelength, _ = extract_wavelength_nm(
            dict(dataset.tags(index)),
            dict(dataset.tags(index, ns="IMAGERY")),
            description,
        )
        descriptions.append(description)
        wavelengths.append(wavelength)

    rpc_tags = {str(key): str(value) for key, value in dataset.tags(ns="RPC").items()}
    return RasterMetadata(
        crs_wkt=_crs_wkt(dataset, path),
        transform=transform,
        nodata=_nodata_for(dataset, indices, path),
        band_descriptions=tuple(descriptions),
        wavelengths_nm=tuple(wavelengths),
        dataset_tags={str(key): str(value) for key, value in dataset.tags().items()},
        rpc_tags=rpc_tags or None,
    )


def _to_rasterio_window(window: ReadWindow) -> Window:
    """Convert the pure value object into the rasterio type used by the library call."""
    return Window(
        col_off=window.col_off,
        row_off=window.row_off,
        width=window.width,
        height=window.height,
    )


def _read_from_dataset(
    dataset: Any,
    path: Path,
    window: ReadWindow,
    indices: tuple[int, ...],
    options: ReadOptions,
) -> RasterChunk:
    """Read one window from an already-open dataset and wrap it in a chunk."""
    dtype = options.output_dtype or _widest_dtype(dataset, indices)
    estimate = estimate_read_bytes(window, len(indices), dtype, masked=options.masked)
    _check_budget(estimate, options, path, window)

    rio_window = _to_rasterio_window(window)
    try:
        data: npt.NDArray[Any] = dataset.read(
            indexes=list(indices),
            window=rio_window,
            masked=options.masked,
            out_dtype=options.output_dtype,
        )
    except (RasterioIOError, MemoryError) as error:
        raise RasterReadError(
            "Could not read the requested window.",
            hint="The file may be truncated or the block it needs may be corrupt; "
            "`hypersat inspect` reports the layout the driver sees.",
            context={
                "path": str(path),
                "window": [window.col_off, window.row_off, window.width, window.height],
                "bands": list(indices),
                "reason": str(error),
            },
        ) from error

    if options.fill_nan:
        data = np.ma.filled(data, np.nan)

    return RasterChunk(
        data=data,
        band_indices=indices,
        window=window,
        source_width=int(dataset.width),
        source_height=int(dataset.height),
        metadata=_chunk_metadata(dataset, indices, rio_window, path),
    )


def read_chunk(
    path: Path,
    *,
    window: ReadWindow | None = None,
    bands: Sequence[int] | None = None,
    options: ReadOptions | None = None,
) -> RasterChunk:
    """Read a window of selected bands into memory.

    Args:
        path: Raster to read.
        window: Region to read; defaults to the whole raster, which the memory guard will
            usually refuse for a real hyperspectral product.
        bands: One-based band indices, in the order they should appear in the result.
            ``None`` reads every band in file order.
        options: How to materialise the pixels; defaults to a masked read of the source
            dtype under the default budget.

    Returns:
        A :class:`~hypersat.models.raster.RasterChunk` carrying the pixels and the
        metadata needed to write them back out.

    Raises:
        RasterReadError: If the raster cannot be opened or read, or the window or band
            selection is invalid.
        MemoryBudgetExceededError: If the result would exceed the configured budget.
    """
    effective = options if options is not None else ReadOptions()
    with open_raster(path) as dataset:
        indices = _validated_band_indices(int(dataset.count), bands)
        read_window = _validated_window(dataset, window, path)
        chunk = _read_from_dataset(dataset, path, read_window, indices, effective)

    logger.debug(
        "read raster window",
        extra={
            "path": str(path),
            "window": [
                read_window.col_off,
                read_window.row_off,
                read_window.width,
                read_window.height,
            ],
            "bands": list(chunk.band_indices),
            "dtype": str(chunk.data.dtype),
            "masked": chunk.is_masked,
        },
    )
    return chunk


def iter_block_windows(path: Path, *, band: int = 1) -> Iterator[ReadWindow]:
    """Yield the raster's native block windows, in the order they are stored.

    Following the file's own blocking matters: GDAL decompresses a whole block to serve
    any pixel inside it, so a chunking that straddles block boundaries reads the same
    bytes several times.

    Args:
        path: Raster to walk.
        band: One-based band whose block layout is used. Bands of one dataset normally
            share a layout; the parameter exists because GDAL does not guarantee it.

    Yields:
        One :class:`~hypersat.models.raster.ReadWindow` per block.

    Raises:
        RasterReadError: If the raster cannot be opened or the band does not exist.
    """
    with open_raster(path) as dataset:
        _validated_band_indices(int(dataset.count), [band])
        for _, block in dataset.block_windows(band):
            yield ReadWindow(
                col_off=int(block.col_off),
                row_off=int(block.row_off),
                width=int(block.width),
                height=int(block.height),
            )


def iter_chunks(
    path: Path,
    *,
    bands: Sequence[int] | None = None,
    options: ReadOptions | None = None,
) -> Iterator[RasterChunk]:
    """Read a raster block by block, keeping peak memory to one block at a time.

    The dataset is opened once for the whole walk rather than per block, which matters for
    a file whose directory is expensive to parse.

    Args:
        path: Raster to read.
        bands: One-based band indices, in output order. ``None`` reads every band.
        options: How to materialise each block; the budget applies per block.

    Yields:
        One chunk per block of the first selected band.

    Raises:
        RasterReadError: If the raster cannot be opened or read.
        MemoryBudgetExceededError: If a single block would exceed the budget.
    """
    effective = options if options is not None else ReadOptions()
    with open_raster(path) as dataset:
        indices = _validated_band_indices(int(dataset.count), bands)
        for _, block in dataset.block_windows(indices[0]):
            window = ReadWindow(
                col_off=int(block.col_off),
                row_off=int(block.row_off),
                width=int(block.width),
                height=int(block.height),
            )
            yield _read_from_dataset(dataset, path, window, indices, effective)
