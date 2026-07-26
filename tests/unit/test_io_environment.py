"""Tests for geospatial runtime diagnostics."""

from __future__ import annotations

import pytest
from rasterio import env as rasterio_env

from hypersat.io import environment as env_module
from hypersat.io.environment import (
    bundled_proj_data_path,
    describe_environment,
    ensure_usable_proj_data,
    gdal_bindings_version,
    is_proj_database_usable,
)
from hypersat.models.environment import ProjDataStatus


def test_describe_environment_reports_versions() -> None:
    info = describe_environment()

    assert info.rasterio_version
    assert info.gdal_version
    assert info.proj_version
    assert info.hypersat_version


def test_gdal_bindings_are_optional() -> None:
    # The `gdal` extra is not required; the report only states whether it is present.
    version = gdal_bindings_version()

    assert version is None or isinstance(version, str)


def test_bundled_proj_database_ships_with_rasterio() -> None:
    bundled = bundled_proj_data_path()

    assert bundled is not None
    assert (bundled / "proj.db").is_file()


def test_proj_database_is_usable_in_the_test_session() -> None:
    # The session fixture repairs a hijacked PROJ_LIB, so this must hold afterwards.
    assert is_proj_database_usable() is True
    assert ensure_usable_proj_data() is ProjDataStatus.OK


def _always_broken() -> bool:
    """Stand in for a PROJ database that never answers an EPSG lookup."""
    return False


def _ignore_search_path(path: str) -> None:
    """Stand in for a fallback that has no effect."""


def test_broken_proj_database_is_repaired_from_the_bundled_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[str] = []

    def probe() -> bool:
        # Broken until the bundled search path has been applied, usable afterwards.
        return bool(applied)

    monkeypatch.setattr(env_module, "is_proj_database_usable", probe)
    monkeypatch.setattr(rasterio_env, "set_proj_data_search_path", applied.append)

    status = ensure_usable_proj_data(allow_repair=True)

    assert status is ProjDataStatus.REPAIRED
    assert len(applied) == 1
    assert applied[0].endswith("proj_data")


def test_repair_can_be_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module, "is_proj_database_usable", _always_broken)

    assert ensure_usable_proj_data(allow_repair=False) is ProjDataStatus.BROKEN


def test_unfixable_proj_database_is_reported_as_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module, "is_proj_database_usable", _always_broken)
    monkeypatch.setattr(rasterio_env, "set_proj_data_search_path", _ignore_search_path)

    assert ensure_usable_proj_data(allow_repair=True) is ProjDataStatus.BROKEN


def test_environment_serialises_to_json() -> None:
    payload = describe_environment(ProjDataStatus.REPAIRED).to_json_dict()

    assert payload["proj_data_status"] == "repaired"
    assert isinstance(payload["bundled_proj_data"], str)
