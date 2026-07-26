"""Model describing the geospatial runtime the pipeline is executing on."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from hypersat.models.base import StrictModel

__all__ = ["EnvironmentInfo", "ProjDataStatus"]


class ProjDataStatus(StrEnum):
    """Outcome of checking whether the PROJ coordinate database is usable.

    A frequent Windows failure mode: another product (PostGIS, QGIS, an old OSGeo4W)
    exports a machine-wide ``PROJ_LIB``/``PROJ_DATA`` pointing at an outdated ``proj.db``,
    which overrides the database bundled in the rasterio wheel. Every CRS lookup then
    fails with a confusing ``CRSError``.
    """

    OK = "ok"
    """The configured PROJ database answered an EPSG lookup successfully."""

    REPAIRED = "repaired"
    """The configured database was unusable; rasterio's bundled database is now in use."""

    BROKEN = "broken"
    """No usable PROJ database was found. CRS handling will not work."""

    UNCHECKED = "unchecked"
    """No lookup was attempted."""


class EnvironmentInfo(StrictModel):
    """Versions and data-path configuration of the geospatial stack."""

    hypersat_version: str
    python_version: str
    platform: str
    rasterio_version: str
    gdal_version: str
    """GDAL version of the library rasterio is linked against."""
    proj_version: str
    geos_version: str | None = None
    proj_data_status: ProjDataStatus = ProjDataStatus.UNCHECKED
    proj_lib_env: str | None = None
    """Value of ``PROJ_LIB``/``PROJ_DATA`` in the environment, if set."""
    bundled_proj_data: Path | None = None
    gdal_bindings_version: str | None = None
    """Version of the optional ``osgeo.gdal`` bindings, or ``None`` when not installed."""
