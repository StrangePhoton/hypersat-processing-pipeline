"""Helpers that write tiny synthetic rasters for tests.

**These are test inputs, never processing results.** The pixel values are arithmetic
patterns and the RPC coefficients below are an identity-like placeholder, not a real
sensor model. Nothing produced here represents real satellite data, and no test may
present its output as a satellite-processing result (see ``docs/data-sources.md``).

Every raster is a few pixels across so the whole suite stays fast and needs no committed
binary products.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, from_origin

__all__ = [
    "SYNTHETIC_RPC_TAGS",
    "write_dem",
    "write_enmap_like_product",
    "write_geotiff",
]


def _rpc_coeff_string(term_index: int | None = None) -> str:
    """Build a 20-term RPC coefficient string.

    When ``term_index`` is set, that term is ``1.0`` and the rest are zero. When ``None``,
    the constant term is ``1.0`` (a unit denominator).
    """
    coefficients = [0.0] * 20
    coefficients[0 if term_index is None else term_index] = 1.0
    return " ".join(str(value) for value in coefficients)


SYNTHETIC_RPC_TAGS: dict[str, str] = {
    "LINE_OFF": "3.0",
    "SAMP_OFF": "4.0",
    "LAT_OFF": "45.0",
    "LONG_OFF": "10.0",
    "HEIGHT_OFF": "300.0",
    "LINE_SCALE": "3.0",
    "SAMP_SCALE": "4.0",
    "LAT_SCALE": "0.05",
    "LONG_SCALE": "0.05",
    "HEIGHT_SCALE": "500.0",
    # Invertible toy model: normalised row = lat_n, normalised column = lon_n.
    # Coefficient order is RPC00B (constant, L, P, H, ...). Enough for software tests of
    # warping; not a real sensor model and not suitable for accuracy claims.
    "LINE_NUM_COEFF": _rpc_coeff_string(2),
    "LINE_DEN_COEFF": _rpc_coeff_string(),
    "SAMP_NUM_COEFF": _rpc_coeff_string(1),
    "SAMP_DEN_COEFF": _rpc_coeff_string(),
}
"""A structurally complete but scientifically meaningless RPC model.

It is sufficient to test *software* behaviour - detection, completeness checks, error
paths and that the warp machinery runs - and is explicitly not sufficient to validate
geometric accuracy.
"""


def write_geotiff(
    path: Path,
    *,
    width: int = 8,
    height: int = 6,
    count: int = 3,
    dtype: str = "uint16",
    crs: str | None = "EPSG:32633",
    transform: Affine | None = None,
    nodata: float | None = 0.0,
    band_descriptions: Sequence[str] | None = None,
    wavelengths_nm: Sequence[float] | None = None,
    wavelength_units: str | None = "nm",
    rpc_tags: Mapping[str, str] | None = None,
    dataset_tags: Mapping[str, str] | None = None,
    tiled: bool = False,
    compress: str | None = None,
    fill_value: float | None = None,
    nodata_pixels: Sequence[tuple[int, int]] | None = None,
) -> Path:
    """Write a small synthetic GeoTIFF and return its path.

    Args:
        path: Destination file; parent directories are created.
        width: Raster width in pixels.
        height: Raster height in pixels.
        count: Number of bands.
        dtype: NumPy dtype name.
        crs: CRS string, or ``None`` to write a raster without a CRS.
        transform: Affine transform; defaults to a 30 m grid, or identity when ``crs`` is
            ``None`` so the raster mimics sensor geometry.
        nodata: NoData value, or ``None`` to leave it unset.
        band_descriptions: Per-band description strings.
        wavelengths_nm: Per-band centre wavelengths written as band tags.
        wavelength_units: Value of the ``wavelength_units`` band tag. ``None`` omits it,
            which exercises the unit-inference path.
        rpc_tags: RPC metadata to write into the ``RPC`` domain.
        dataset_tags: Tags for the default dataset metadata domain.
        tiled: Whether to write tiled rather than striped data.
        compress: GDAL compression name, e.g. ``"deflate"``.
        fill_value: Constant to fill every band with; defaults to a per-band ramp.
        nodata_pixels: ``(row, column)`` positions set to ``nodata`` in every band, so a
            raster actually contains the value its metadata declares.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = (
            Affine.identity() if crs is None else from_origin(500000.0, 5000000.0, 30.0, 30.0)
        )

    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "transform": transform,
        "tiled": tiled,
    }
    if crs is not None:
        profile["crs"] = CRS.from_string(crs)
    if nodata is not None:
        profile["nodata"] = nodata
    if tiled:
        profile["blockxsize"] = 16
        profile["blockysize"] = 16
    if compress is not None:
        profile["compress"] = compress

    # Writing an identity transform is deliberate: it is how a sensor-geometry product
    # looks. rasterio warns on every such write, so the expected warning is silenced here to
    # keep the suite's output readable. The filter must be installed before `open`, which is
    # what warns. The production reader records the condition as a field instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dataset:
            for band_index in range(1, count + 1):
                if fill_value is None:
                    ramp = np.arange(width * height, dtype=dtype).reshape(height, width)
                    values = ramp + band_index
                else:
                    values = np.full((height, width), fill_value, dtype=dtype)
                if nodata_pixels and nodata is not None:
                    for row, column in nodata_pixels:
                        values[row, column] = nodata
                dataset.write(values, band_index)
                if band_descriptions is not None:
                    dataset.set_band_description(band_index, band_descriptions[band_index - 1])
                if wavelengths_nm is not None:
                    tags = {"wavelength": str(wavelengths_nm[band_index - 1])}
                    if wavelength_units is not None:
                        tags["wavelength_units"] = wavelength_units
                    dataset.update_tags(band_index, **tags)
            if rpc_tags:
                dataset.update_tags(ns="RPC", **dict(rpc_tags))
            if dataset_tags:
                dataset.update_tags(**dict(dataset_tags))
    return path


def write_dem(
    path: Path,
    *,
    width: int = 10,
    height: int = 10,
    crs: str | None = "EPSG:4326",
    nodata: float | None = -32768.0,
    count: int = 1,
) -> Path:
    """Write a small synthetic single-band elevation raster.

    Args:
        path: Destination file.
        width: Raster width in pixels.
        height: Raster height in pixels.
        crs: CRS string, or ``None`` for a DEM without georeferencing.
        nodata: NoData value, or ``None`` to leave it unset.
        count: Band count; more than one exercises the "unexpected DEM shape" warning.

    Returns:
        The path written.
    """
    transform = Affine.identity() if crs is None else from_origin(9.9, 45.1, 0.01, 0.01)
    return write_geotiff(
        path,
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        fill_value=300.0,
    )


def write_enmap_like_product(
    root: Path,
    *,
    with_rpc: bool = True,
    band_count: int = 4,
    extra_raster: bool = False,
) -> Path:
    """Write a directory that mimics the *layout* of an EnMAP L1B product.

    Only the layout is imitated: an imagery file named ``*-SPECTRAL_IMAGE.TIF`` beside a
    metadata XML file. The XML is a stub, and nothing here implements or claims to
    implement EnMAP's product specification.

    Args:
        root: Product directory to create.
        with_rpc: Whether the imagery carries an RPC sensor model.
        band_count: Number of synthetic bands.
        extra_raster: Also write an unrelated raster, to exercise ambiguity handling.

    Returns:
        The product directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    stem = "ENMAP01-____L1B-DT0000000000_20260101T000000Z_001_V010000_20260101T000000Z"
    wavelengths = [420.0 + 100.0 * index for index in range(band_count)]
    write_geotiff(
        root / f"{stem}-SPECTRAL_IMAGE.TIF",
        width=8,
        height=6,
        count=band_count,
        dtype="uint16",
        crs=None,
        nodata=0.0,
        band_descriptions=[f"band {index + 1}" for index in range(band_count)],
        wavelengths_nm=wavelengths,
        rpc_tags=SYNTHETIC_RPC_TAGS if with_rpc else None,
    )
    (root / f"{stem}-METADATA.XML").write_text(
        "<!-- synthetic stub, not an EnMAP metadata document -->\n<product/>\n",
        encoding="utf-8",
    )
    if extra_raster:
        write_geotiff(root / f"{stem}-QL_SWIR.TIF", width=4, height=4, count=1)
    return root
