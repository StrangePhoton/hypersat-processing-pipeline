"""Build percentile-stretched PNG previews from a raster.

This module is the only place that turns scientific samples into an 8-bit display image.
It reads through :mod:`hypersat.io.reader` (so the memory budget and masked-NoData contract
still apply), stretches in :mod:`hypersat.visualization.stretch`, optionally blurs in
:mod:`hypersat.visualization.preprocess`, and writes a PNG - never a GeoTIFF.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from hypersat.analytics.bands import BandMatch, false_colour_bands, true_colour_bands
from hypersat.exceptions import (
    InvalidWavelengthMetadataError,
    MemoryBudgetExceededError,
    PreviewError,
)
from hypersat.io.environment import ensure_usable_proj_data
from hypersat.io.files import derive_product_id
from hypersat.io.inspect import inspect_raster, resolve_raster_path
from hypersat.io.reader import DEFAULT_READ_BUDGET_BYTES, ReadOptions, read_chunk
from hypersat.logging_config import get_logger
from hypersat.models.config import RGB_BAND_COUNT, PreviewComposite, PreviewRequest
from hypersat.models.product import RasterInfo
from hypersat.models.raster import ReadWindow
from hypersat.processing.validation import validate_output_directory
from hypersat.visualization.preprocess import (
    gaussian_blur,
    target_preview_shape,
)
from hypersat.visualization.stretch import stretch_to_uint8

__all__ = [
    "PreviewComposite",
    "PreviewResult",
    "derive_product_id",
    "render_preview",
    "write_png",
]

logger = get_logger(__name__)

_IMAGE_NDIM_GREY = 2
_IMAGE_NDIM_COLOR = 3


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """Outcome of a successful preview render.

    Attributes:
        path: PNG that was written.
        composite: Composite kind that was produced.
        band_indices: One-based source bands used, in display order.
        wavelengths_nm: Centre wavelengths of those bands, when known.
        width: Preview width in pixels.
        height: Preview height in pixels.
        product_id: Token used in the output filename.
    """

    path: Path
    composite: PreviewComposite
    band_indices: tuple[int, ...]
    wavelengths_nm: tuple[float | None, ...]
    width: int
    height: int
    product_id: str


def write_png(
    path: Path,
    image: npt.NDArray[np.uint8],
    *,
    overwrite: bool = False,
) -> Path:
    """Write an 8-bit image to a PNG atomically.

    Args:
        path: Destination ``.png`` path. The parent directory must already exist.
        image: ``(rows, columns)`` greyscale or ``(rows, columns, 3)`` RGB array.
        overwrite: Whether an existing file may be replaced.

    Returns:
        The path written.

    Raises:
        PreviewError: If the destination is unusable or OpenCV cannot encode the PNG.
    """
    if path.exists() and not overwrite:
        raise PreviewError(
            "The preview file already exists.",
            hint="Pass --overwrite, or choose a different --output-dir / product id.",
            context={"path": str(path)},
        )
    if not path.parent.is_dir():
        raise PreviewError(
            "The output directory does not exist.",
            hint="Create it first, or pass --output-dir and let the command create it.",
            context={"path": str(path), "directory": str(path.parent)},
        )
    if image.ndim == _IMAGE_NDIM_GREY:
        to_write: npt.NDArray[np.uint8] = image
    elif image.ndim == _IMAGE_NDIM_COLOR and image.shape[2] == RGB_BAND_COUNT:
        # OpenCV expects BGR channel order for colour images.
        to_write = cast(
            "npt.NDArray[np.uint8]",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
    else:
        raise PreviewError(
            "PNG previews must be greyscale or 3-channel RGB.",
            hint="Pass an (H, W) or (H, W, 3) uint8 array.",
            context={"shape": [int(size) for size in image.shape]},
        )

    # Encode in memory, then write bytes with pathlib. OpenCV's imwrite uses a narrow
    # fopen on Windows and silently fails on paths that contain non-ASCII characters
    # (common under pytest's per-user temp directory).
    ok, encoded = cv2.imencode(".png", to_write)
    if not ok:
        raise PreviewError(
            "OpenCV could not encode the PNG.",
            hint="Check that the preview array is finite uint8 greyscale or RGB.",
            context={"path": str(path), "shape": [int(size) for size in image.shape]},
        )

    temporary = path.with_name(f".{path.stem}-{os.getpid()}-{id(image):x}.png")
    try:
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PreviewError(
            "Could not write the preview PNG.",
            hint="Check free space and write permission on the destination directory.",
            context={"path": str(path), "reason": str(error)},
        ) from error
    return path


def _composite_label(composite: PreviewComposite, band_indices: Sequence[int]) -> str:
    """Return the filename token for a composite."""
    if composite is PreviewComposite.TRUE_COLOR:
        return "true_color"
    if composite is PreviewComposite.FALSE_COLOR:
        return "false_color"
    return f"band_{band_indices[0]}"


def _resolve_band_indices(
    info: RasterInfo,
    composite: PreviewComposite,
    *,
    bands: Sequence[int] | None,
    band: int | None,
    tolerance_nm: float,
) -> tuple[tuple[int, ...], tuple[float | None, ...]]:
    """Resolve which source bands to read, preserving display order."""
    if bands is not None:
        indices = tuple(bands)
        wavelengths = tuple(
            info.bands[index - 1].wavelength_nm if 1 <= index <= info.band_count else None
            for index in indices
        )
        return indices, wavelengths

    wavelengths_nm = [item.wavelength_nm for item in info.bands]

    if composite is PreviewComposite.BAND:
        if band is None:
            raise PreviewError(
                "A single-band preview needs an explicit --band.",
                hint="Pass --band with a 1-based index, or --composite true-color / "
                "false-color to build a composite from wavelengths.",
                context={"band_count": info.band_count},
            )
        if band < 1 or band > info.band_count:
            raise PreviewError(
                "Requested band is outside the raster's band range.",
                hint=f"This raster has {info.band_count} band(s); indices are 1-based.",
                context={"band": band, "band_count": info.band_count},
            )
        wavelength = info.bands[band - 1].wavelength_nm
        return (band,), (wavelength,)

    try:
        matches: tuple[BandMatch, BandMatch, BandMatch]
        if composite is PreviewComposite.TRUE_COLOR:
            matches = true_colour_bands(wavelengths_nm, tolerance_nm=tolerance_nm)
        else:
            matches = false_colour_bands(wavelengths_nm, tolerance_nm=tolerance_nm)
    except InvalidWavelengthMetadataError as error:
        raise PreviewError(
            str(error.message),
            hint=(
                error.hint
                or "Pass --bands with three 1-based indices, or --band for a single-band preview."
            ),
            context=error.context,
        ) from error

    return (
        tuple(match.index for match in matches),
        tuple(match.wavelength_nm for match in matches),
    )


def _bands_to_hwc(data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Convert a ``(bands, rows, columns)`` cube to an image OpenCV/PNG expect."""
    if data.shape[0] == 1:
        return cast("npt.NDArray[np.uint8]", data[0])
    if data.shape[0] == RGB_BAND_COUNT:
        transposed: npt.NDArray[np.uint8] = np.transpose(data, (1, 2, 0))
        return transposed
    raise PreviewError(
        "Previews support one band or exactly three bands.",
        hint="Use --composite band with --band, or a three-band composite / --bands R,G,B.",
        context={"band_count": int(data.shape[0])},
    )


def render_preview(
    request: PreviewRequest,
    *,
    max_bytes: int | None = DEFAULT_READ_BUDGET_BYTES,
    window: ReadWindow | None = None,
) -> PreviewResult:
    """Render one cosmetic PNG preview of a raster or product directory.

    Large inputs are read already resampled to ``max_dimension`` via GDAL (see
    :class:`~hypersat.io.reader.ReadOptions.out_shape`), so a hyperspectral cube never has
    to enter memory at full resolution just to make a thumbnail.

    Args:
        request: Validated preview configuration.
        max_bytes: Memory budget for the (already downsampled) read.
        window: Optional source window; defaults to the whole raster.

    Returns:
        Metadata about the PNG that was written.

    Raises:
        PreviewError: If band selection, stretching or PNG writing fails.
        RasterReadError: If the raster cannot be opened or read.
        MemoryBudgetExceededError: If even the downsampled read exceeds the budget.
        OutputPathError: If the output directory cannot be created or written to.
    """
    ensure_usable_proj_data(allow_repair=request.proj_autofix)
    validate_output_directory(request.output.directory, overwrite=request.output.overwrite)

    raster_path, _layout = resolve_raster_path(request.product_path)
    info = inspect_raster(raster_path)
    resolved_id = request.product_id or derive_product_id(request.product_path)

    band_indices, wavelengths = _resolve_band_indices(
        info,
        request.composite,
        bands=request.bands,
        band=request.band,
        tolerance_nm=request.wavelength_tolerance_nm,
    )

    source_window = window or ReadWindow.covering(info.width, info.height)
    out_height, out_width = target_preview_shape(
        source_window.width, source_window.height, request.max_dimension
    )
    options = ReadOptions(
        masked=True,
        as_float32=True,
        max_bytes=max_bytes,
        out_shape=(out_height, out_width),
    )

    try:
        chunk = read_chunk(
            raster_path,
            window=source_window,
            bands=band_indices,
            options=options,
        )
    except MemoryBudgetExceededError as error:
        raise MemoryBudgetExceededError(
            error.message,
            hint=(
                "Lower --max-dimension, pass a smaller --window, or raise the read budget. "
                "Previews already downsample while reading; if this still fires the preview "
                "size itself is too large for the configured budget."
            ),
            context=error.context,
        ) from error

    stretched = stretch_to_uint8(
        chunk.data,
        mask=chunk.mask,
        lower_percentile=request.stretch.lower_percentile,
        upper_percentile=request.stretch.upper_percentile,
        per_band=request.stretch.per_band,
    )
    image = _bands_to_hwc(stretched)
    if request.blur_kernel is not None:
        image = cast("npt.NDArray[np.uint8]", gaussian_blur(image, request.blur_kernel))

    label = _composite_label(request.composite, band_indices)
    destination = request.output.directory / f"{resolved_id}_preview_{label}.png"
    write_png(destination, image, overwrite=request.output.overwrite)

    height, width = int(image.shape[0]), int(image.shape[1])
    logger.info(
        "wrote preview",
        extra={
            "path": str(destination),
            "composite": request.composite.value,
            "bands": list(band_indices),
            "width": width,
            "height": height,
        },
    )
    return PreviewResult(
        path=destination,
        composite=request.composite,
        band_indices=band_indices,
        wavelengths_nm=wavelengths,
        width=width,
        height=height,
        product_id=resolved_id,
    )
