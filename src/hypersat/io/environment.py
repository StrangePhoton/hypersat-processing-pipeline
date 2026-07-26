"""Geospatial runtime diagnostics.

Two practical problems motivate this module.

**A hijacked PROJ database.** rasterio's wheels bundle their own ``proj.db``, but a
machine-wide ``PROJ_LIB``/``PROJ_DATA`` exported by another product (PostGIS, QGIS, an old
OSGeo4W installation) takes precedence. If that database is older than the PROJ library
inside the wheel, *every* CRS lookup fails with a message that looks like a bug in this
project. :func:`ensure_usable_proj_data` detects the situation and, unless disabled, points
PROJ back at the bundled database - always with a warning in the log, never silently.

**Optional GDAL bindings.** ``osgeo.gdal`` is an optional extra. Everything the current
milestone needs is available through rasterio, including metadata-domain listing, so the
bindings are reported for transparency rather than required.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from types import ModuleType

import rasterio
from rasterio import env as rasterio_env
from rasterio.crs import CRS
from rasterio.errors import CRSError

from hypersat import __version__
from hypersat.logging_config import get_logger
from hypersat.models.environment import EnvironmentInfo, ProjDataStatus

osgeo_gdal: ModuleType | None
try:  # The `gdal` extra is optional; see docs/data-sources.md.
    from osgeo import gdal as _osgeo_gdal
except ImportError:  # pragma: no cover - depends on the installed extras
    osgeo_gdal = None
else:  # pragma: no cover - depends on the installed extras
    osgeo_gdal = _osgeo_gdal

__all__ = [
    "PROJ_CONFLICT_HINT",
    "bundled_proj_data_path",
    "describe_environment",
    "ensure_usable_proj_data",
    "gdal_bindings_version",
    "is_proj_database_usable",
]

logger = get_logger(__name__)

PROJ_CONFLICT_HINT = (
    "A PROJ_LIB/PROJ_DATA environment variable is pointing at a proj.db from another "
    "installation (PostGIS, QGIS, OSGeo4W). Either unset it for this shell, point it at "
    "the proj_data directory inside the installed rasterio package, or keep the automatic "
    "fallback enabled (--proj-autofix)."
)

_PROBE_EPSG = 4326
"""WGS 84: present in every PROJ database, so a failed lookup means a broken database."""


def gdal_bindings_version() -> str | None:
    """Return the version of the optional ``osgeo.gdal`` bindings, if importable."""
    if osgeo_gdal is None:
        return None
    version = getattr(osgeo_gdal, "__version__", None)
    return str(version) if version is not None else "unknown"


def bundled_proj_data_path() -> Path | None:
    """Return the ``proj_data`` directory shipped inside the rasterio wheel, if any."""
    found = rasterio_env.PROJDataFinder().search()
    return Path(str(found)) if found else None


def is_proj_database_usable() -> bool:
    """Return whether the active PROJ database can resolve a well-known EPSG code."""
    try:
        CRS.from_epsg(_PROBE_EPSG)
    except CRSError:
        return False
    return True


def ensure_usable_proj_data(*, allow_repair: bool = True) -> ProjDataStatus:
    """Verify the PROJ database and, if broken, fall back to rasterio's bundled copy.

    Changing ``os.environ`` is not enough once PROJ has initialised, so the fallback uses
    ``rasterio.env.set_proj_data_search_path``, which updates PROJ's search path in place.

    Args:
        allow_repair: When false, a broken database is reported but left untouched.

    Returns:
        The resulting :class:`~hypersat.models.environment.ProjDataStatus`.
    """
    if is_proj_database_usable():
        return ProjDataStatus.OK

    configured = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if not allow_repair:
        logger.warning(
            "PROJ database is unusable and automatic repair is disabled",
            extra={"proj_env": configured, "hint": PROJ_CONFLICT_HINT},
        )
        return ProjDataStatus.BROKEN

    bundled = bundled_proj_data_path()
    if bundled is None:
        logger.error(
            "PROJ database is unusable and no bundled database was found",
            extra={"proj_env": configured, "hint": PROJ_CONFLICT_HINT},
        )
        return ProjDataStatus.BROKEN

    rasterio_env.set_proj_data_search_path(str(bundled))
    if not is_proj_database_usable():
        logger.error(
            "PROJ database is still unusable after falling back to the bundled copy",
            extra={"bundled_proj_data": str(bundled), "hint": PROJ_CONFLICT_HINT},
        )
        return ProjDataStatus.BROKEN

    logger.warning(
        "replaced an unusable PROJ database with the copy bundled in rasterio",
        extra={
            "proj_env": configured,
            "bundled_proj_data": str(bundled),
            "hint": PROJ_CONFLICT_HINT,
        },
    )
    return ProjDataStatus.REPAIRED


def describe_environment(
    proj_data_status: ProjDataStatus = ProjDataStatus.UNCHECKED,
) -> EnvironmentInfo:
    """Collect versions and PROJ configuration of the running geospatial stack.

    Args:
        proj_data_status: Result of a previous :func:`ensure_usable_proj_data` call, so the
            report shows whether the fallback was needed.

    Returns:
        A populated :class:`~hypersat.models.environment.EnvironmentInfo`.
    """
    return EnvironmentInfo(
        hypersat_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        rasterio_version=str(rasterio.__version__),
        gdal_version=str(rasterio.__gdal_version__),
        proj_version=str(rasterio.__proj_version__),
        geos_version=str(getattr(rasterio, "__geos_version__", "")) or None,
        proj_data_status=proj_data_status,
        proj_lib_env=os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"),
        bundled_proj_data=bundled_proj_data_path(),
        gdal_bindings_version=gdal_bindings_version(),
    )
