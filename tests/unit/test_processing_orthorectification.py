"""Unit tests for RPC + DEM orthorectification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hypersat.exceptions import (
    MissingDEMError,
    MissingRPCMetadataError,
    UnreadableDEMError,
)
from hypersat.io.inspect import inspect_raster
from hypersat.models.config import (
    DataSemantics,
    OrthorectifyRequest,
    OutputConfig,
    ResamplingMethod,
)
from hypersat.processing.orthorectification import (
    orthorectify_raster,
    rpc_geographic_bounds,
    validate_dem_covers_scene,
)
from tests.support.rasters import SYNTHETIC_RPC_TAGS, write_geotiff


def _covering_dem(path: Path) -> Path:
    """DEM covering the synthetic RPC normalisation domain around lon=10, lat=45."""
    return write_geotiff(
        path,
        width=40,
        height=40,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(9.9, 45.1, 0.005, 0.005),
        nodata=-32768.0,
        fill_value=300.0,
    )


def _sensor_product(path: Path, *, with_rpc: bool = True) -> Path:
    return write_geotiff(
        path,
        width=8,
        height=6,
        count=2,
        crs=None,
        nodata=0.0,
        fill_value=100,
        wavelengths_nm=(560.0, 665.0),
        rpc_tags=SYNTHETIC_RPC_TAGS if with_rpc else None,
    )


def test_rpc_geographic_bounds_from_offsets(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif")
    rpc = inspect_raster(product).rpc

    left, bottom, right, top = rpc_geographic_bounds(rpc)

    assert left == pytest.approx(9.95)
    assert right == pytest.approx(10.05)
    assert bottom == pytest.approx(44.95)
    assert top == pytest.approx(45.05)


def test_dem_coverage_rejects_non_overlapping_dem(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif")
    dem = write_geotiff(
        tmp_path / "dem.tif",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
        nodata=-32768.0,
        fill_value=300.0,
    )
    scene = rpc_geographic_bounds(inspect_raster(product).rpc)
    dem_info = inspect_raster(dem)

    with pytest.raises(UnreadableDEMError, match="does not overlap"):
        validate_dem_covers_scene(dem_info, scene)


def test_orthorectify_writes_map_geometry_geotiff(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif")
    dem = _covering_dem(tmp_path / "dem.tif")

    result = orthorectify_raster(
        OrthorectifyRequest(
            product_path=product,
            dem_path=dem,
            output=OutputConfig(directory=tmp_path / "out"),
            target_crs="EPSG:32632",
            resolution=60.0,
        )
    )

    assert result.path.exists()
    assert "ortho_epsg32632_60m" in result.path.name
    assert result.dem_path == dem
    assert result.transformer_options["RPC_DEM"] == str(dem)
    assert result.resampling is ResamplingMethod.BILINEAR
    with rasterio.open(result.path) as dataset:
        assert dataset.crs.to_epsg() == 32632
        assert dataset.count == 2
        assert abs(dataset.transform.a) == pytest.approx(60.0)
        data = np.array(dataset.read(1), copy=True)
        assert np.count_nonzero(data) > 0


def test_orthorectify_auto_selects_utm(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif")
    dem = _covering_dem(tmp_path / "dem.tif")

    result = orthorectify_raster(
        OrthorectifyRequest(
            product_path=product,
            dem_path=dem,
            output=OutputConfig(directory=tmp_path / "out"),
            target_crs="auto",
            resolution=100.0,
            bands=(1,),
        )
    )

    assert result.crs_authority == "EPSG:32632"
    assert result.band_indices == (1,)


def test_missing_rpc_is_refused(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif", with_rpc=False)
    dem = _covering_dem(tmp_path / "dem.tif")

    with pytest.raises(MissingRPCMetadataError, match="RPC"):
        orthorectify_raster(
            OrthorectifyRequest(
                product_path=product,
                dem_path=dem,
                output=OutputConfig(directory=tmp_path / "out"),
                target_crs="EPSG:32632",
            )
        )


def test_missing_dem_is_refused(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif")

    with pytest.raises(MissingDEMError):
        orthorectify_raster(
            OrthorectifyRequest(
                product_path=product,
                dem_path=tmp_path / "absent.tif",
                output=OutputConfig(directory=tmp_path / "out"),
                target_crs="EPSG:32632",
            )
        )


def test_categorical_forces_nearest(tmp_path: Path) -> None:
    product = _sensor_product(tmp_path / "src.tif")
    dem = _covering_dem(tmp_path / "dem.tif")

    result = orthorectify_raster(
        OrthorectifyRequest(
            product_path=product,
            dem_path=dem,
            output=OutputConfig(directory=tmp_path / "out"),
            target_crs="EPSG:32632",
            resolution=90.0,
            data_semantics=DataSemantics.CATEGORICAL,
            resampling=ResamplingMethod.BILINEAR,
            bands=(1,),
        )
    )

    assert result.resampling is ResamplingMethod.NEAREST
