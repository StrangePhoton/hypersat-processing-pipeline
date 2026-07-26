"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypersat.io.environment import ensure_usable_proj_data
from hypersat.models.environment import ProjDataStatus
from tests.support.rasters import (
    SYNTHETIC_RPC_TAGS,
    write_dem,
    write_enmap_like_product,
    write_geotiff,
)


@pytest.fixture(scope="session", autouse=True)
def usable_proj_database() -> ProjDataStatus:
    """Make sure EPSG lookups work before any fixture writes a georeferenced raster.

    A machine-wide ``PROJ_LIB``/``PROJ_DATA`` from another product (PostGIS, QGIS,
    OSGeo4W) overrides the PROJ database bundled in the rasterio wheel and breaks every
    CRS lookup. This is a property of the developer's machine, not of the code under test,
    so the suite repairs it explicitly rather than failing everywhere with ``CRSError``.
    The production code path is exercised directly in ``tests/unit/test_io_environment.py``.
    """
    return ensure_usable_proj_data(allow_repair=True)


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return an empty temporary directory that stands in for a product/output root."""
    return tmp_path


@pytest.fixture
def sample_raster(tmp_path: Path) -> Path:
    """A small georeferenced multi-band GeoTIFF with wavelengths and descriptions."""
    return write_geotiff(
        tmp_path / "scene.tif",
        count=3,
        band_descriptions=["blue", "green", "red"],
        wavelengths_nm=[490.0, 560.0, 665.0],
        dataset_tags={"SENSOR": "synthetic"},
    )


@pytest.fixture
def sensor_geometry_raster(tmp_path: Path) -> Path:
    """A raster mimicking L1B input: no CRS or transform, but an RPC sensor model."""
    return write_geotiff(
        tmp_path / "l1b_like.tif",
        crs=None,
        count=4,
        wavelengths_nm=[420.0, 560.0, 665.0, 842.0],
        rpc_tags=SYNTHETIC_RPC_TAGS,
    )


@pytest.fixture
def product_directory(tmp_path: Path) -> Path:
    """A directory imitating the layout of an EnMAP L1B product."""
    return write_enmap_like_product(tmp_path / "product")


@pytest.fixture
def dem_raster(tmp_path: Path) -> Path:
    """A small georeferenced synthetic elevation raster."""
    return write_dem(tmp_path / "dem.tif")
