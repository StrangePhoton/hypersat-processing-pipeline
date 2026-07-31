"""Tests for windowed, band-selective reading and the memory guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypersat.exceptions import MemoryBudgetExceededError, RasterReadError
from hypersat.io.reader import (
    ReadOptions,
    estimate_read_bytes,
    iter_block_windows,
    iter_chunks,
    read_chunk,
)
from hypersat.models.raster import ReadWindow


def test_a_full_read_returns_every_band_in_file_order(sample_raster: Path) -> None:
    chunk = read_chunk(sample_raster)

    assert chunk.band_indices == (1, 2, 3)
    assert chunk.data.shape == (3, 6, 8)
    assert chunk.covers_full_grid is True


def test_band_selection_preserves_the_requested_order(sample_raster: Path) -> None:
    # Order is the point: a composite asks for (red, green, blue), not for sorted bands.
    chunk = read_chunk(sample_raster, bands=[3, 1])

    assert chunk.band_indices == (3, 1)
    assert chunk.metadata.band_descriptions == ("red", "blue")
    np.testing.assert_array_equal(chunk.band(3), chunk.data[0])


def test_a_window_reads_only_that_region(sample_raster: Path) -> None:
    window = ReadWindow(col_off=2, row_off=1, width=3, height=2)

    chunk = read_chunk(sample_raster, window=window, bands=[1])

    assert chunk.data.shape == (1, 2, 3)
    assert chunk.covers_full_grid is False
    full = read_chunk(sample_raster, bands=[1])
    np.testing.assert_array_equal(chunk.data[0], full.data[0][1:3, 2:5])


def test_a_window_shifts_the_transform_to_its_own_origin(sample_raster: Path) -> None:
    full = read_chunk(sample_raster, bands=[1])
    window = ReadWindow(col_off=2, row_off=1, width=3, height=2)

    chunk = read_chunk(sample_raster, window=window, bands=[1])

    assert full.metadata.transform is not None
    assert chunk.metadata.transform is not None
    pixel_size_x, _, origin_x, _, pixel_size_y, origin_y = full.metadata.transform
    assert chunk.metadata.transform[2] == pytest.approx(origin_x + 2 * pixel_size_x)
    assert chunk.metadata.transform[5] == pytest.approx(origin_y + 1 * pixel_size_y)


def test_sensor_geometry_input_reports_no_transform(sensor_geometry_raster: Path) -> None:
    # An identity transform means "not on a map grid". Shifting it for a window would
    # invent a translation, so the reader reports None instead.
    chunk = read_chunk(sensor_geometry_raster, window=ReadWindow(1, 1, 2, 2), bands=[1])

    assert chunk.metadata.transform is None
    assert chunk.metadata.crs_wkt is None


def test_nodata_samples_are_masked_rather_than_zeroed(masked_raster: Path) -> None:
    chunk = read_chunk(masked_raster, bands=[1])

    assert chunk.is_masked is True
    assert bool(chunk.mask[0, 0, 0]) is True
    assert bool(chunk.mask[0, 2, 3]) is True
    assert int(chunk.masked.count()) == 6 * 4 - 2


def test_reading_unmasked_returns_a_plain_array(masked_raster: Path) -> None:
    chunk = read_chunk(masked_raster, bands=[1], options=ReadOptions(masked=False))

    assert chunk.is_masked is False
    assert chunk.data[0, 0, 0] == 0
    with pytest.raises(TypeError, match="ReadOptions\\(masked=True\\)"):
        _ = chunk.masked


def test_float32_conversion_keeps_the_mask(masked_raster: Path) -> None:
    chunk = read_chunk(masked_raster, bands=[1], options=ReadOptions(as_float32=True))

    assert chunk.data.dtype == np.float32
    assert chunk.is_masked is True


def test_nan_filling_is_opt_in_and_produces_a_plain_float_array(masked_raster: Path) -> None:
    chunk = read_chunk(masked_raster, bands=[1], options=ReadOptions(fill_nan=True))

    assert chunk.is_masked is False
    assert chunk.data.dtype == np.float32
    assert bool(np.isnan(chunk.data[0, 0, 0])) is True
    assert int(np.count_nonzero(np.isnan(chunk.data))) == 2


def test_nan_filling_without_a_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="fill_nan requires masked=True"):
        ReadOptions(masked=False, fill_nan=True)


def test_a_non_positive_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        ReadOptions(max_bytes=0)


def test_the_estimate_accounts_for_dtype_band_count_and_mask() -> None:
    window = ReadWindow(0, 0, 10, 10)

    assert estimate_read_bytes(window, 3, "uint16") == 100 * 3 * 2
    assert estimate_read_bytes(window, 3, "float32") == 100 * 3 * 4
    assert estimate_read_bytes(window, 3, "uint16", masked=True) == 100 * 3 * 3


def test_an_oversized_read_is_refused_before_allocating(sample_raster: Path) -> None:
    with pytest.raises(MemoryBudgetExceededError) as excinfo:
        read_chunk(sample_raster, options=ReadOptions(max_bytes=16))

    error = excinfo.value
    assert error.exit_code == 5
    assert error.context["budget_bytes"] == 16
    assert error.context["estimate_bytes"] > 16
    assert "iter_chunks" in (error.hint or "")


def test_the_budget_can_be_disabled_deliberately(sample_raster: Path) -> None:
    chunk = read_chunk(sample_raster, options=ReadOptions(max_bytes=None))

    assert chunk.band_count == 3


def test_a_window_outside_the_raster_is_rejected(sample_raster: Path) -> None:
    with pytest.raises(RasterReadError, match="extends beyond the raster"):
        read_chunk(sample_raster, window=ReadWindow(col_off=6, row_off=0, width=8, height=2))


def test_band_indices_outside_the_raster_are_rejected(sample_raster: Path) -> None:
    with pytest.raises(RasterReadError) as excinfo:
        read_chunk(sample_raster, bands=[1, 99])

    assert excinfo.value.context["out_of_range"] == [99]


def test_an_empty_band_selection_is_rejected(sample_raster: Path) -> None:
    with pytest.raises(RasterReadError, match="No bands were requested"):
        read_chunk(sample_raster, bands=[])


def test_a_missing_file_raises_a_read_error(tmp_path: Path) -> None:
    with pytest.raises(RasterReadError, match="Could not open the raster"):
        read_chunk(tmp_path / "absent.tif")


def test_a_corrupt_file_raises_a_read_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"II*\x00 this is not a TIFF")

    with pytest.raises(RasterReadError):
        read_chunk(corrupt)


def test_wavelengths_and_descriptions_travel_with_the_pixels(sample_raster: Path) -> None:
    chunk = read_chunk(sample_raster)

    assert chunk.metadata.wavelengths_nm == (490.0, 560.0, 665.0)
    assert chunk.metadata.band_descriptions == ("blue", "green", "red")
    assert chunk.metadata.nodata == 0.0
    assert chunk.metadata.crs_wkt is not None


def test_the_rpc_domain_is_captured_for_later_decisions(sensor_geometry_raster: Path) -> None:
    chunk = read_chunk(sensor_geometry_raster, bands=[1])

    assert chunk.metadata.rpc_tags is not None
    assert "LINE_OFF" in chunk.metadata.rpc_tags


def test_block_windows_tile_the_raster_exactly_once(tiled_raster: Path) -> None:
    windows = list(iter_block_windows(tiled_raster))

    assert len(windows) == 9  # 40 px over a 16 px tile grid: 3 x 3, the last row/column partial
    covered = np.zeros((40, 40), dtype=np.uint8)
    for window in windows:
        covered[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] += 1
    assert covered.min() == 1
    assert covered.max() == 1


def test_iterating_chunks_reads_each_block_once(tiled_raster: Path) -> None:
    chunks = list(iter_chunks(tiled_raster, bands=[1]))
    whole = read_chunk(tiled_raster, bands=[1])

    assert len(chunks) == 9
    reassembled = np.zeros((40, 40), dtype=whole.data.dtype)
    for chunk in chunks:
        window = chunk.window
        reassembled[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] = chunk.data[0]
    np.testing.assert_array_equal(reassembled, whole.data[0])


def test_the_budget_applies_per_block_while_iterating(tiled_raster: Path) -> None:
    with pytest.raises(MemoryBudgetExceededError):
        list(iter_chunks(tiled_raster, bands=[1], options=ReadOptions(max_bytes=8)))


def test_block_iteration_validates_the_band(tiled_raster: Path) -> None:
    with pytest.raises(RasterReadError):
        list(iter_block_windows(tiled_raster, band=99))
