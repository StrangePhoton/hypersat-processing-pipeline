"""Tests for atomic GeoTIFF writing and metadata propagation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from hypersat.exceptions import RasterWriteError
from hypersat.io import writer
from hypersat.io.inspect import inspect_raster
from hypersat.io.reader import ReadOptions, read_chunk
from hypersat.io.writer import Compression, GeoTiffOptions, write_array, write_chunk
from hypersat.models.raster import RasterMetadata, ReadWindow

_WGS84_UTM33N = (
    'PROJCS["WGS 84 / UTM zone 33N",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],'
    'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","32633"]]'
)


def _ramp(bands: int = 2, rows: int = 4, columns: int = 5) -> np.ndarray:
    return np.arange(bands * rows * columns, dtype="uint16").reshape(bands, rows, columns)


def _georeferenced(nodata: float | None = None) -> RasterMetadata:
    """Metadata for a raster on a 30 m UTM grid, so outputs are not sensor-geometry."""
    return RasterMetadata(
        crs_wkt=_WGS84_UTM33N,
        transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5000000.0),
        nodata=nodata,
    )


def test_metadata_survives_a_write(tmp_path: Path) -> None:
    destination = tmp_path / "out.tif"
    metadata = RasterMetadata(
        crs_wkt=_WGS84_UTM33N,
        transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 5000000.0),
        nodata=0.0,
        band_descriptions=("green", "red"),
        wavelengths_nm=(560.0, 665.0),
        dataset_tags={"SENSOR": "synthetic"},
    )

    write_array(destination, _ramp(), metadata=metadata)

    info = inspect_raster(destination)
    assert info.crs.epsg == 32633
    assert info.nodata == 0.0
    assert info.transform[:6] == [30.0, 0.0, 500000.0, 0.0, -30.0, 5000000.0]
    assert [band.description for band in info.bands] == ["green", "red"]
    assert [band.wavelength_nm for band in info.bands] == [560.0, 665.0]
    assert info.metadata["SENSOR"] == "synthetic"


def test_the_default_layout_is_tiled_and_compressed(tmp_path: Path) -> None:
    destination = tmp_path / "out.tif"

    write_array(destination, _ramp(rows=300, columns=300), metadata=_georeferenced())

    with rasterio.open(destination) as dataset:
        assert dataset.profile["tiled"] is True
        assert dataset.profile["blockxsize"] == 256
        assert dataset.compression.name.lower() == "deflate"


def test_a_two_dimensional_array_is_written_as_one_band(tmp_path: Path) -> None:
    destination = tmp_path / "single.tif"

    write_array(destination, np.ones((4, 5), dtype="uint8"))

    assert inspect_raster(destination).band_count == 1


def test_masked_samples_are_written_as_nodata(tmp_path: Path) -> None:
    destination = tmp_path / "masked.tif"
    mask = np.zeros((1, 4, 5), dtype=bool)
    mask[0, 0, 0] = True
    data = np.ma.masked_array(_ramp(bands=1), mask=mask)

    write_array(destination, data, metadata=_georeferenced(nodata=9999.0))

    written = read_chunk(destination, options=ReadOptions(masked=False))
    assert written.data[0, 0, 0] == 9999


def test_masked_data_without_a_nodata_value_is_refused(tmp_path: Path) -> None:
    data = np.ma.masked_array(_ramp(bands=1), mask=np.ones((1, 4, 5), dtype=bool))

    with pytest.raises(RasterWriteError, match="without a NoData value"):
        write_array(tmp_path / "out.tif", data)


def test_an_existing_file_is_not_silently_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "out.tif"
    write_array(destination, _ramp())

    with pytest.raises(RasterWriteError, match="already exists"):
        write_array(destination, _ramp())

    write_array(destination, _ramp(bands=1), overwrite=True)
    assert inspect_raster(destination).band_count == 1


def test_a_missing_output_directory_is_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(RasterWriteError) as excinfo:
        write_array(tmp_path / "absent" / "out.tif", _ramp())

    assert "hypersat validate" in (excinfo.value.hint or "")


def test_a_successful_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    write_array(tmp_path / "out.tif", _ramp())

    assert [entry.name for entry in tmp_path.iterdir()] == ["out.tif"]


def test_a_crash_mid_write_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates the case the temporary file exists for: the dataset was created, bytes
    # were written, and the process failed before the file was complete.
    def explode(destination: Path, *_args: object, **_kwargs: object) -> None:
        destination.write_bytes(b"half a GeoTIFF")
        raise RasterWriteError("simulated failure part-way through writing")

    monkeypatch.setattr(writer, "_write_dataset", explode)

    with pytest.raises(RasterWriteError, match="simulated failure"):
        write_array(tmp_path / "out.tif", _ramp())

    assert list(tmp_path.iterdir()) == []


def test_an_unusable_crs_is_reported_rather_than_written(tmp_path: Path) -> None:
    with pytest.raises(RasterWriteError, match="coordinate reference system"):
        write_array(
            tmp_path / "out.tif",
            _ramp(),
            metadata=RasterMetadata(crs_wkt="definitely not wkt"),
        )

    assert list(tmp_path.iterdir()) == []


def test_arrays_of_the_wrong_rank_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(RasterWriteError, match="2- or 3-dimensional"):
        write_array(tmp_path / "out.tif", np.arange(5, dtype="uint8"))


def test_block_size_must_suit_the_tiff_specification() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        GeoTiffOptions(block_size=100)


def test_the_bigtiff_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="bigtiff must be one of"):
        GeoTiffOptions(bigtiff="MAYBE")


def test_the_predictor_follows_the_dtype() -> None:
    options = GeoTiffOptions()

    assert options.predictor_for(np.dtype("uint16")) == 2
    assert options.predictor_for(np.dtype("float32")) == 3
    assert GeoTiffOptions(compression=Compression.NONE).predictor_for(np.dtype("float32")) == 1
    assert GeoTiffOptions(predictor=2).predictor_for(np.dtype("float32")) == 2


def test_an_uncompressed_layout_can_be_requested(tmp_path: Path) -> None:
    destination = tmp_path / "raw.tif"

    write_array(
        destination,
        _ramp(),
        metadata=_georeferenced(),
        options=GeoTiffOptions(compression=Compression.NONE, tiled=False),
    )

    with rasterio.open(destination) as dataset:
        assert dataset.compression is None
        assert dataset.profile["tiled"] is False


def test_writing_a_full_grid_chunk_keeps_the_sensor_model(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    chunk = read_chunk(sensor_geometry_raster)

    write_chunk(tmp_path / "copy.tif", chunk)

    info = inspect_raster(tmp_path / "copy.tif")
    assert info.rpc.available is True
    assert info.rpc.is_usable is True


def test_writing_a_windowed_chunk_drops_the_sensor_model(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    # The coefficients are tied to the source line/sample origin, so a subset must not
    # advertise a sensor model it no longer satisfies.
    chunk = read_chunk(sensor_geometry_raster, window=ReadWindow(2, 1, 3, 2))

    write_chunk(tmp_path / "subset.tif", chunk)

    assert inspect_raster(tmp_path / "subset.tif").rpc.available is False


def test_the_sensor_model_can_be_dropped_explicitly(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    chunk = read_chunk(sensor_geometry_raster)

    write_chunk(tmp_path / "no_rpc.tif", chunk, keep_rpc=False)

    assert inspect_raster(tmp_path / "no_rpc.tif").rpc.available is False


def test_a_float32_chunk_round_trips_through_the_writer(
    masked_raster: Path, tmp_path: Path
) -> None:
    chunk = read_chunk(masked_raster, options=ReadOptions(as_float32=True))

    write_chunk(tmp_path / "float.tif", chunk)

    info = inspect_raster(tmp_path / "float.tif")
    assert info.dtypes == ["float32", "float32"]
    assert info.nodata == 0.0
