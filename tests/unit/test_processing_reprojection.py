"""Unit tests for CRS reprojection and grid alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hypersat.exceptions import MissingGeoreferencingError, ReprojectionError
from hypersat.models.config import (
    DataSemantics,
    OutputConfig,
    ReprojectRequest,
    ResamplingMethod,
)
from hypersat.processing.reprojection import (
    format_resolution_token,
    reproject_raster,
    resampling_for_semantics,
    select_utm_epsg,
)
from tests.support.rasters import SYNTHETIC_RPC_TAGS, write_geotiff


def test_select_utm_epsg_northern_and_southern() -> None:
    assert select_utm_epsg(10.0, 45.0) == 32632
    assert select_utm_epsg(10.0, -45.0) == 32732


def test_select_utm_epsg_rejects_polar_latitudes() -> None:
    with pytest.raises(ValueError, match="UTM domain"):
        select_utm_epsg(0.0, 85.0)


def test_format_resolution_token() -> None:
    assert format_resolution_token(30.0) == "30m"
    assert format_resolution_token(0.5) == "0p5m"


def test_categorical_semantics_force_nearest() -> None:
    assert (
        resampling_for_semantics(DataSemantics.CATEGORICAL, ResamplingMethod.BILINEAR)
        is ResamplingMethod.NEAREST
    )
    assert (
        resampling_for_semantics(DataSemantics.CONTINUOUS, ResamplingMethod.CUBIC)
        is ResamplingMethod.CUBIC
    )


def test_reproject_request_forces_nearest_for_categorical() -> None:
    request = ReprojectRequest(
        product_path=Path("scene.tif"),
        output=OutputConfig(directory=Path("out")),
        target_crs="EPSG:32633",
        data_semantics=DataSemantics.CATEGORICAL,
        resampling=ResamplingMethod.BILINEAR,
    )

    assert request.resampling is ResamplingMethod.NEAREST


def test_reproject_writes_map_geometry_geotiff(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "src.tif",
        width=8,
        height=6,
        count=2,
        crs="EPSG:32633",
        transform=from_origin(500000.0, 5000000.0, 30.0, 30.0),
        wavelengths_nm=(560.0, 665.0),
    )

    result = reproject_raster(
        ReprojectRequest(
            product_path=source,
            output=OutputConfig(directory=tmp_path / "out"),
            target_crs="EPSG:32633",
            resolution=60.0,
            snap_to_grid=True,
        )
    )

    assert result.path.exists()
    assert "reprojected_epsg32633_60m" in result.path.name
    assert result.resolution == pytest.approx(60.0)
    assert result.snapped is True
    with rasterio.open(result.path) as dataset:
        assert dataset.crs.to_epsg() == 32633
        assert dataset.count == 2
        assert abs(dataset.transform.a) == pytest.approx(60.0)
        # Snapped origins land on a multiple of the resolution.
        assert dataset.transform.c % 60.0 == pytest.approx(0.0)


def test_reproject_auto_selects_utm(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "wgs84.tif",
        width=10,
        height=10,
        count=1,
        crs="EPSG:4326",
        transform=from_origin(10.0, 45.1, 0.01, 0.01),
        nodata=None,
    )

    result = reproject_raster(
        ReprojectRequest(
            product_path=source,
            output=OutputConfig(directory=tmp_path / "out"),
            target_crs="auto",
            resolution=100.0,
        )
    )

    assert result.crs_authority == "EPSG:32632"
    assert "epsg32632" in result.path.name


def test_align_to_reference_grid(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "src.tif",
        width=8,
        height=6,
        count=1,
        crs="EPSG:32633",
        transform=from_origin(500015.0, 5000015.0, 30.0, 30.0),
        fill_value=42,
    )
    reference = write_geotiff(
        tmp_path / "ref.tif",
        width=20,
        height=20,
        count=1,
        crs="EPSG:32633",
        transform=from_origin(500000.0, 5001000.0, 60.0, 60.0),
        fill_value=1,
    )

    result = reproject_raster(
        ReprojectRequest(
            product_path=source,
            output=OutputConfig(directory=tmp_path / "out"),
            reference_raster=reference,
            resampling=ResamplingMethod.NEAREST,
        )
    )

    assert "aligned_epsg32633_60m" in result.path.name
    assert result.reference_raster == reference
    with rasterio.open(result.path) as dataset:
        assert abs(dataset.transform.a) == pytest.approx(60.0)
        assert dataset.transform.c == pytest.approx(500000.0)


def test_sensor_geometry_is_refused(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "l1b.tif",
        crs=None,
        count=1,
        rpc_tags=SYNTHETIC_RPC_TAGS,
    )

    with pytest.raises(MissingGeoreferencingError, match="affine"):
        reproject_raster(
            ReprojectRequest(
                product_path=source,
                output=OutputConfig(directory=tmp_path / "out"),
                target_crs="EPSG:32633",
                resolution=30.0,
            )
        )


def test_non_overlapping_reference_fails(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "src.tif",
        width=4,
        height=4,
        count=1,
        crs="EPSG:32633",
        transform=from_origin(500000.0, 5000000.0, 30.0, 30.0),
    )
    reference = write_geotiff(
        tmp_path / "ref.tif",
        width=4,
        height=4,
        count=1,
        crs="EPSG:32633",
        transform=from_origin(600000.0, 6000000.0, 30.0, 30.0),
    )

    with pytest.raises(ReprojectionError, match="does not overlap"):
        reproject_raster(
            ReprojectRequest(
                product_path=source,
                output=OutputConfig(directory=tmp_path / "out"),
                reference_raster=reference,
            )
        )


def test_band_subset_is_honoured(tmp_path: Path) -> None:
    source = write_geotiff(
        tmp_path / "src.tif",
        width=6,
        height=6,
        count=3,
        crs="EPSG:32633",
        wavelengths_nm=(490.0, 560.0, 665.0),
    )

    result = reproject_raster(
        ReprojectRequest(
            product_path=source,
            output=OutputConfig(directory=tmp_path / "out"),
            target_crs="EPSG:32633",
            resolution=30.0,
            bands=(2,),
        )
    )

    assert result.band_indices == (2,)
    with rasterio.open(result.path) as dataset:
        assert dataset.count == 1
        tags = dataset.tags(1)
        assert float(tags["wavelength"]) == pytest.approx(560.0)
        data = np.array(dataset.read(1), copy=True)
        assert data.shape == (dataset.height, dataset.width)
