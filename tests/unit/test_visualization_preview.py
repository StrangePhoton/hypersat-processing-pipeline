"""Tests for PNG preview rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hypersat.exceptions import PreviewError
from hypersat.models.config import (
    OutputConfig,
    PreviewComposite,
    PreviewRequest,
    StretchConfig,
)
from hypersat.visualization.preview import derive_product_id, render_preview, write_png
from tests.support.rasters import write_geotiff


def _request(
    input_path: Path,
    output_dir: Path,
    *,
    composite: PreviewComposite = PreviewComposite.TRUE_COLOR,
    bands: tuple[int, ...] | None = None,
    band: int | None = None,
    max_dimension: int = 64,
    blur_kernel: int | None = None,
    product_id: str | None = None,
    overwrite: bool = False,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> PreviewRequest:
    return PreviewRequest(
        product_path=input_path,
        output=OutputConfig(directory=output_dir, overwrite=overwrite),
        composite=composite,
        bands=bands,
        band=band,
        stretch=StretchConfig(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        ),
        max_dimension=max_dimension,
        blur_kernel=blur_kernel,
        product_id=product_id,
    )


def test_derive_product_id_slugifies_the_stem() -> None:
    assert derive_product_id(Path("EnMAP-L1B_Scene.tif")) == "enmap_l1b_scene"


def test_true_color_preview_selects_bands_by_wavelength(
    sample_raster: Path, tmp_path: Path
) -> None:
    result = render_preview(_request(sample_raster, tmp_path))

    assert result.path.is_file()
    assert result.path.name == "scene_preview_true_color.png"
    assert result.band_indices == (3, 2, 1)
    assert result.wavelengths_nm == (665.0, 560.0, 490.0)
    image = _read_png(result.path)
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_false_color_preview_uses_nir_red_green(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    result = render_preview(
        _request(sensor_geometry_raster, tmp_path, composite=PreviewComposite.FALSE_COLOR)
    )

    assert result.band_indices == (4, 3, 2)
    assert result.path.name.endswith("preview_false_color.png")


def test_single_band_preview_is_greyscale(sample_raster: Path, tmp_path: Path) -> None:
    result = render_preview(
        _request(sample_raster, tmp_path, composite=PreviewComposite.BAND, band=2)
    )

    assert result.band_indices == (2,)
    assert result.path.name == "scene_preview_band_2.png"
    image = _read_png(result.path)
    assert image.ndim == 2


def test_explicit_bands_preserve_display_order(sample_raster: Path, tmp_path: Path) -> None:
    result = render_preview(
        _request(
            sample_raster,
            tmp_path,
            composite=PreviewComposite.TRUE_COLOR,
            bands=(1, 3, 2),
        )
    )

    assert result.band_indices == (1, 3, 2)


def test_nodata_pixels_become_black_in_the_png(masked_raster: Path, tmp_path: Path) -> None:
    result = render_preview(
        _request(
            masked_raster,
            tmp_path,
            composite=PreviewComposite.BAND,
            band=1,
            lower_percentile=0.0,
            upper_percentile=100.0,
        )
    )

    image = _read_png(result.path)
    assert int(image[0, 0]) == 0


def test_max_dimension_downsamples_while_reading(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "large.tif",
        width=120,
        height=80,
        count=3,
        wavelengths_nm=[490.0, 560.0, 665.0],
    )

    result = render_preview(_request(source, tmp_path / "out", max_dimension=40))

    assert max(result.width, result.height) == 40
    assert result.height in {26, 27}  # rounding of 80 * 40/120


def test_blur_kernel_is_applied(sample_raster: Path, tmp_path: Path) -> None:
    sharp = render_preview(_request(sample_raster, tmp_path / "sharp", product_id="sharp"))
    soft = render_preview(
        _request(
            sample_raster,
            tmp_path / "soft",
            blur_kernel=5,
            product_id="soft",
        )
    )

    assert not np.array_equal(_read_png(sharp.path), _read_png(soft.path))


def test_existing_preview_is_refused_without_overwrite(sample_raster: Path, tmp_path: Path) -> None:
    render_preview(_request(sample_raster, tmp_path, max_dimension=32))

    with pytest.raises(PreviewError, match="already exists"):
        render_preview(_request(sample_raster, tmp_path, max_dimension=32))


def test_write_png_rejects_wrong_channel_count(tmp_path: Path) -> None:
    with pytest.raises(PreviewError, match="greyscale or 3-channel"):
        write_png(tmp_path / "bad.png", np.zeros((4, 4, 2), dtype=np.uint8))


def _read_png(path: Path) -> np.ndarray:
    # imdecode avoids OpenCV's narrow-path imread, which fails on non-ASCII temp dirs.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
