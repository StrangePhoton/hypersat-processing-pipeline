"""End-to-end read, select and write flow over generated fixtures.

The fixtures are synthetic rasters, so these tests prove that *metadata is preserved and
geometry claims stay honest*. They prove nothing about the geometric accuracy of any real
satellite product.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from hypersat.analytics.bands import true_colour_bands
from hypersat.io.inspect import inspect_raster
from hypersat.io.reader import ReadOptions, read_chunk
from hypersat.io.writer import write_chunk
from hypersat.models.raster import ReadWindow

pytestmark = pytest.mark.integration


def test_a_windowed_subset_stays_georeferenced_at_its_own_origin(
    sample_raster: Path, tmp_path: Path
) -> None:
    source = inspect_raster(sample_raster)
    window = ReadWindow(col_off=2, row_off=1, width=4, height=3)

    chunk = read_chunk(sample_raster, window=window, bands=[1, 2])
    destination = write_chunk(tmp_path / "subset.tif", chunk)

    result = inspect_raster(destination)
    assert (result.width, result.height, result.band_count) == (4, 3, 2)
    assert result.crs.epsg == source.crs.epsg
    assert source.bounds is not None
    assert result.bounds is not None
    # Two columns in from the left edge and one row down, at 30 m pixels.
    assert result.bounds[0] == pytest.approx(source.bounds[0] + 2 * 30.0)
    assert result.bounds[3] == pytest.approx(source.bounds[3] - 1 * 30.0)


def test_wavelength_selection_drives_the_bands_that_get_written(
    sample_raster: Path, tmp_path: Path
) -> None:
    source = inspect_raster(sample_raster)
    wavelengths = [band.wavelength_nm for band in source.bands]

    red, green, blue = true_colour_bands(wavelengths)
    chunk = read_chunk(sample_raster, bands=[red.index, green.index, blue.index])
    destination = write_chunk(tmp_path / "composite.tif", chunk)

    result = inspect_raster(destination)
    assert [band.wavelength_nm for band in result.bands] == [665.0, 560.0, 490.0]
    assert [band.description for band in result.bands] == ["red", "green", "blue"]


def test_a_full_copy_of_sensor_geometry_input_keeps_its_sensor_model(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    chunk = read_chunk(sensor_geometry_raster)
    destination = write_chunk(tmp_path / "copy.tif", chunk)

    result = inspect_raster(destination)
    assert result.rpc.available is True
    assert result.rpc.is_usable is True
    assert result.has_affine_georeferencing is False
    assert result.rpc.line_off == 3.0


def test_a_subset_of_sensor_geometry_input_loses_it_with_a_warning(
    sensor_geometry_raster: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    chunk = read_chunk(sensor_geometry_raster, window=ReadWindow(1, 1, 3, 3))

    with caplog.at_level(logging.WARNING, logger="hypersat.io.writer"):
        destination = write_chunk(tmp_path / "subset.tif", chunk)

    assert "dropping the RPC sensor model" in caplog.text
    assert inspect_raster(destination).rpc.available is False


def test_pixels_survive_the_round_trip_unchanged(masked_raster: Path, tmp_path: Path) -> None:
    chunk = read_chunk(masked_raster)
    destination = write_chunk(tmp_path / "roundtrip.tif", chunk)

    reread = read_chunk(destination)
    np.testing.assert_array_equal(reread.mask, chunk.mask)
    np.testing.assert_array_equal(reread.masked.filled(0), chunk.masked.filled(0))


def test_block_by_block_reading_matches_a_single_read(tiled_raster: Path, tmp_path: Path) -> None:
    whole = read_chunk(tiled_raster, bands=[1], options=ReadOptions(masked=False))
    destination = write_chunk(tmp_path / "copy.tif", whole)

    reread = read_chunk(destination, bands=[1], options=ReadOptions(masked=False))
    np.testing.assert_array_equal(reread.data, whole.data)
