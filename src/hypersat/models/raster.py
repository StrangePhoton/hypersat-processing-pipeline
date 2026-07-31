"""In-memory raster structures shared by the reader, the writer and processing code.

Unlike everything else in :mod:`hypersat.models`, these are frozen dataclasses rather than
Pydantic models. The reason is :class:`RasterChunk`: it carries a NumPy array, so Pydantic
would need ``arbitrary_types_allowed`` and still could not serialise it to JSON, which is
the property every other model here has. Keeping pixels out of the Pydantic layer also
keeps the quality-control report serialisable by construction.

Nothing in this module imports rasterio, so processing functions that consume a chunk can
be tested with hand-built arrays and no raster on disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = [
    "AFFINE_COEFFICIENT_COUNT",
    "RasterChunk",
    "RasterMetadata",
    "ReadWindow",
]

AFFINE_COEFFICIENT_COUNT = 6
"""An affine transform is stored as ``(a, b, c, d, e, f)``, in rasterio's order."""

_EXPECTED_NDIM = 3


@dataclass(frozen=True, slots=True)
class ReadWindow:
    """A rectangular region of a raster, in pixel coordinates.

    Offsets are zero-based and measured from the top-left corner, matching GDAL and
    rasterio. This is a plain value object: converting it to a ``rasterio.windows.Window``
    happens in :mod:`hypersat.io.reader`, so that the models layer stays free of geospatial
    imports.

    Attributes:
        col_off: Zero-based column of the left edge.
        row_off: Zero-based row of the top edge.
        width: Width in pixels; strictly positive.
        height: Height in pixels; strictly positive.
    """

    col_off: int
    row_off: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject windows that cannot describe a real region.

        Raises:
            ValueError: If an offset is negative or a dimension is not positive.
        """
        if self.col_off < 0 or self.row_off < 0:
            raise ValueError(
                f"window offsets must not be negative, got "
                f"col_off={self.col_off}, row_off={self.row_off}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"window dimensions must be positive, got width={self.width}, height={self.height}"
            )

    @classmethod
    def covering(cls, width: int, height: int) -> ReadWindow:
        """Return the window covering an entire raster of the given size.

        Args:
            width: Raster width in pixels.
            height: Raster height in pixels.

        Returns:
            A window anchored at the origin spanning the whole grid.
        """
        return cls(col_off=0, row_off=0, width=width, height=height)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``, matching the shape of one band's array."""
        return self.height, self.width

    @property
    def pixel_count(self) -> int:
        """Return the number of pixels in one band of this window."""
        return self.width * self.height

    def fits_within(self, width: int, height: int) -> bool:
        """Report whether the window lies entirely inside a raster of the given size.

        Args:
            width: Raster width in pixels.
            height: Raster height in pixels.

        Returns:
            ``True`` when no part of the window falls outside the raster.
        """
        return self.col_off + self.width <= width and self.row_off + self.height <= height


@dataclass(frozen=True, slots=True)
class RasterMetadata:
    """Everything needed to write pixels back out as a georeferenced raster.

    The CRS is held as WKT rather than as a ``rasterio.crs.CRS`` so that this layer stays
    import-free; the writer parses it. ``transform`` is ``None`` for imagery in sensor
    geometry, which has no affine mapping to the ground - an EnMAP L1B product is exactly
    that case.

    Attributes:
        crs_wkt: Coordinate reference system as WKT, or ``None`` if undefined.
        transform: Six affine coefficients ``(a, b, c, d, e, f)``, or ``None``.
        nodata: NoData value written into the output, or ``None`` to leave it unset.
        band_descriptions: Per-band description strings; entries may be ``None``.
        wavelengths_nm: Per-band centre wavelengths in nanometres; entries may be ``None``.
        dataset_tags: Tags for the default dataset metadata domain.
        rpc_tags: RPC sensor-model tags. Only valid for output on the *original* pixel
            grid: the coefficients are expressed against the source line/sample origin, so
            attaching them to a windowed subset would describe the wrong geometry.
    """

    crs_wkt: str | None = None
    transform: tuple[float, ...] | None = None
    nodata: float | None = None
    band_descriptions: tuple[str | None, ...] = ()
    wavelengths_nm: tuple[float | None, ...] = ()
    dataset_tags: Mapping[str, str] = field(default_factory=dict)
    rpc_tags: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate the affine transform's length.

        Raises:
            ValueError: If ``transform`` does not hold exactly six coefficients.
        """
        if self.transform is not None and len(self.transform) != AFFINE_COEFFICIENT_COUNT:
            raise ValueError(
                f"transform must have {AFFINE_COEFFICIENT_COUNT} coefficients, "
                f"got {len(self.transform)}"
            )


@dataclass(frozen=True, slots=True)
class RasterChunk:
    """Pixels read from a raster, together with the metadata needed to interpret them.

    ``data`` is always three-dimensional, ``(bands, rows, columns)``, even for a single
    band, so downstream code never has to branch on dimensionality. It may be a
    ``numpy.ma.MaskedArray``, which is what the reader returns by default: NoData is
    metadata rather than a magic number, and a mask keeps it that way instead of letting a
    fill value take part in arithmetic.

    Attributes:
        data: Pixel array shaped ``(bands, rows, columns)``.
        band_indices: One-based source band indices, in the order they appear in ``data``.
        window: The region of the source raster the pixels came from.
        source_width: Width of the source raster, used to detect a full-grid read.
        source_height: Height of the source raster.
        metadata: Georeferencing and per-band metadata for the pixels in this chunk.
    """

    data: npt.NDArray[Any]
    band_indices: tuple[int, ...]
    window: ReadWindow
    source_width: int
    source_height: int
    metadata: RasterMetadata

    def __post_init__(self) -> None:
        """Check that the array, the band list and the window agree.

        Raises:
            ValueError: If the array is not three-dimensional, or its shape contradicts
                the band indices or the window.
        """
        if self.data.ndim != _EXPECTED_NDIM:
            raise ValueError(
                f"data must be 3-dimensional (bands, rows, columns), got {self.data.ndim}"
            )
        bands, rows, columns = self.data.shape
        if bands != len(self.band_indices):
            raise ValueError(
                f"data has {bands} band(s) but {len(self.band_indices)} band index/indices "
                "were given"
            )
        if (rows, columns) != self.window.shape:
            raise ValueError(
                f"data shape {(rows, columns)} does not match window shape {self.window.shape}"
            )

    @property
    def band_count(self) -> int:
        """Return the number of bands held in this chunk."""
        return len(self.band_indices)

    @property
    def is_masked(self) -> bool:
        """Report whether ``data`` is a masked array."""
        return bool(np.ma.isMaskedArray(self.data))

    @property
    def masked(self) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
        """Return the pixels as a masked array.

        ``data`` is typed as a plain array because a chunk may hold either kind. Code that
        needs the mask - anything doing arithmetic that must not consume NoData - goes
        through this property, which makes the requirement explicit and type-checked
        instead of relying on the array happening to be masked.

        Returns:
            The same array, typed as a masked array.

        Raises:
            TypeError: If this chunk holds a plain array.
        """
        if not self.is_masked:
            raise TypeError(
                "this chunk holds a plain array; read it with ReadOptions(masked=True) "
                "to get a mask"
            )
        return cast("np.ma.MaskedArray[Any, np.dtype[Any]]", self.data)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Return the boolean mask, always as a full array.

        NumPy stores the mask of an array with nothing masked as the scalar ``nomask``,
        which indexes and broadcasts differently from a real mask array. Expanding it here
        means callers can treat ``chunk.mask[band, row, column]`` as always valid.

        Returns:
            A ``(bands, rows, columns)`` array that is ``True`` where samples are NoData.

        Raises:
            TypeError: If this chunk holds a plain array.
        """
        return np.ma.getmaskarray(self.masked)

    @property
    def covers_full_grid(self) -> bool:
        """Report whether the window spans the whole source raster.

        A sensor model stays valid only on the grid it was solved for, so this is what the
        writer checks before carrying RPC coefficients into an output file. Selecting a
        subset of *bands* does not affect it: RPC coefficients map pixel coordinates to
        ground coordinates and say nothing about the spectral dimension.
        """
        return (
            self.window.col_off == 0
            and self.window.row_off == 0
            and self.window.width == self.source_width
            and self.window.height == self.source_height
        )

    def band(self, index: int) -> npt.NDArray[Any]:
        """Return the two-dimensional array of one source band.

        Args:
            index: One-based source band index; must be present in ``band_indices``.

        Returns:
            The ``(rows, columns)`` array for that band.

        Raises:
            KeyError: If the band is not part of this chunk.
        """
        try:
            position = self.band_indices.index(index)
        except ValueError as error:
            raise KeyError(
                f"band {index} is not in this chunk; it holds bands {list(self.band_indices)}"
            ) from error
        result: npt.NDArray[Any] = self.data[position]
        return result

    def with_data(self, data: npt.NDArray[Any], bands: Sequence[int] | None = None) -> RasterChunk:
        """Return a copy of this chunk carrying different pixels.

        Used by processing steps that transform values but keep the geometry, so the
        window and georeferencing cannot drift away from the array by accident.

        Args:
            data: Replacement array, shaped ``(bands, rows, columns)``.
            bands: One-based band indices for the new array. Defaults to the current ones.

        Returns:
            A new chunk; this instance is unchanged.
        """
        return RasterChunk(
            data=data,
            band_indices=tuple(bands) if bands is not None else self.band_indices,
            window=self.window,
            source_width=self.source_width,
            source_height=self.source_height,
            metadata=self.metadata,
        )
