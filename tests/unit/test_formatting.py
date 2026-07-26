"""Tests for the human-readable renderers."""

from __future__ import annotations

from pathlib import Path

from hypersat.formatting import render_inspection, render_validation_report
from hypersat.io.inspect import inspect_input
from hypersat.models.validation import CheckStatus, ValidationCheck, ValidationReport
from tests.support.rasters import write_geotiff


def test_inspection_rendering_contains_the_key_facts(sample_raster: Path) -> None:
    text = render_inspection(inspect_input(sample_raster))

    for expected in (
        "Input",
        "Raster",
        "Georeferencing",
        "Storage",
        "Metadata",
        "Bands (3)",
        "EPSG:32633",
        "8 x 6 px",
        "490.00 nm",
    ):
        assert expected in text


def test_band_table_is_truncated_and_says_so(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "cube.tif", count=30, width=4, height=4)

    text = render_inspection(inspect_input(path), band_limit=5)

    assert "Bands (5 of 30 shown)" in text
    assert "25 more band(s)" in text


def test_band_limit_zero_prints_every_band(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "cube.tif", count=12, width=4, height=4)

    text = render_inspection(inspect_input(path), band_limit=0)

    assert "Bands (12)" in text
    assert "more band(s)" not in text


def test_sensor_geometry_rendering_states_the_absence_of_an_affine(
    sensor_geometry_raster: Path,
) -> None:
    text = render_inspection(inspect_input(sensor_geometry_raster))

    assert "affine georeferencing    no" in text
    assert "available (usable)" in text
    assert "Warnings" in text


def test_validation_rendering_shows_status_labels_and_hints() -> None:
    report = ValidationReport(
        input_path=Path("scene.tif"),
        resolved_raster_path=Path("scene.tif"),
        checks=[
            ValidationCheck(name="raster_readable", status=CheckStatus.PASSED, message="ok"),
            ValidationCheck(
                name="rpc_metadata",
                status=CheckStatus.FAILED,
                message="no sensor model",
                hint="use an L1B product",
            ),
            ValidationCheck(
                name="nodata_configured", status=CheckStatus.WARNING, message="unset", hint="set it"
            ),
            ValidationCheck(name="dem", status=CheckStatus.SKIPPED, message="no DEM"),
        ],
    )

    text = render_validation_report(report)

    assert "Validation: INVALID (1 passed, 1 failed, 1 warning, 1 skipped)" in text
    assert "[PASS] raster_readable" in text
    assert "[FAIL] rpc_metadata" in text
    assert "[WARN] nodata_configured" in text
    assert "[SKIP] dem" in text
    assert "hint: use an L1B product" in text


def test_hints_are_not_printed_for_passing_checks() -> None:
    report = ValidationReport(
        input_path=Path("scene.tif"),
        checks=[
            ValidationCheck(
                name="ok_check", status=CheckStatus.PASSED, message="fine", hint="not shown"
            )
        ],
    )

    assert "not shown" not in render_validation_report(report)
