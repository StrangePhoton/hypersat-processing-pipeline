"""Tests for the CLI surface: help text, command output and error-to-exit-code mapping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypersat import __version__
from hypersat.cli import app, main
from hypersat.exceptions import (
    MissingDEMError,
    MissingRPCMetadataError,
    ProductStructureError,
    ProductValidationError,
    RasterReadError,
)
from tests.support.rasters import write_geotiff


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Invoke the console-script entry point and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["hypersat", *argv])
    with pytest.raises(SystemExit) as raised:
        main()
    assert isinstance(raised.value.code, int)
    return raised.value.code


# --------------------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------------------


def test_help_lists_the_implemented_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("inspect", "validate", "version", "preview"):
        assert command in result.output


def test_help_does_not_advertise_unimplemented_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])

    for planned in ("orthorectify", "calculate-index", "process", "spectral-profile"):
        assert planned not in result.output


def test_version_reports_the_package_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_json_is_machine_readable(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["version"] == __version__
    assert payload["name"] == "hypersat-processing-pipeline"


def test_verbose_version_reports_the_geospatial_stack(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version", "--verbose", "--json"])

    payload = json.loads(result.stdout)
    assert payload["environment"]["rasterio_version"]
    assert payload["environment"]["gdal_version"]
    assert "gdal_bindings_version" in payload["environment"]


def test_unknown_log_level_is_a_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--log-level", "LOUD", "version"])

    assert result.exit_code == 2


# --------------------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------------------


def test_inspect_prints_a_readable_report(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(sample_raster)])

    assert result.exit_code == 0, result.output
    assert "EPSG:32633" in result.output
    assert "8 x 6 px" in result.output
    assert "rpc sensor model" in result.output
    assert "rasterio" in result.output


def test_inspect_json_output_parses(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(sample_raster), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["raster"]["driver"] == "GTiff"
    assert payload["raster"]["band_count"] == 3
    assert len(payload["raster"]["bands"]) == 3


def test_inspect_honours_a_band_subset(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(sample_raster), "--bands", "1,3"])

    assert result.exit_code == 0
    assert "Bands (2 of 3 shown)" in result.output


def test_inspect_rejects_a_non_numeric_band_list(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(sample_raster), "--bands", "1,red"])

    assert result.exit_code == 2


def test_inspect_reports_a_checksum_on_request(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(sample_raster), "--checksum"])

    assert "sha256" in result.output


def test_inspect_of_a_product_directory_reports_the_sensor_model(
    runner: CliRunner, product_directory: Path
) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(product_directory)])

    assert result.exit_code == 0
    assert "available (usable)" in result.output
    assert "SPECTRAL_IMAGE" in result.output


def test_inspect_rejects_a_missing_input_path(runner: CliRunner, workspace: Path) -> None:
    result = runner.invoke(app, ["inspect", "--input", str(workspace / "absent")])

    assert isinstance(result.exception, ProductStructureError)
    assert result.exception.exit_code == 4


def test_inspect_reports_a_corrupt_file_as_a_read_error(runner: CliRunner, workspace: Path) -> None:
    corrupt = workspace / "corrupt.tif"
    corrupt.write_bytes(b"not a raster")

    result = runner.invoke(app, ["inspect", "--input", str(corrupt)])

    assert isinstance(result.exception, RasterReadError)


# --------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------


def test_validate_reports_a_valid_product(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["validate", "--input", str(sample_raster)])

    assert result.exit_code == 0, result.output
    assert "Validation: VALID" in result.output
    assert "[PASS] raster_readable" in result.output


def test_validate_json_carries_the_verdict(runner: CliRunner, sample_raster: Path) -> None:
    result = runner.invoke(app, ["validate", "--input", str(sample_raster), "--json"])

    payload = json.loads(result.stdout)
    assert payload["is_valid"] is True
    assert payload["summary"]["failed"] == 0
    assert any(check["name"] == "wavelength_metadata" for check in payload["checks"])


def test_validate_fails_when_a_required_sensor_model_is_absent(
    runner: CliRunner, sample_raster: Path
) -> None:
    result = runner.invoke(app, ["validate", "--input", str(sample_raster), "--require-rpc"])

    assert isinstance(result.exception, MissingRPCMetadataError)
    assert "[FAIL] rpc_metadata" in result.output


def test_validate_accepts_a_sensor_geometry_product_with_a_dem(
    runner: CliRunner, sensor_geometry_raster: Path, dem_raster: Path, workspace: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "validate",
            "--input",
            str(sensor_geometry_raster),
            "--require-rpc",
            "--require-wavelengths",
            "--dem",
            str(dem_raster),
            "--output-dir",
            str(workspace / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Validation: VALID" in result.output
    assert (workspace / "outputs").is_dir()


def test_validate_reports_a_missing_dem(
    runner: CliRunner, sensor_geometry_raster: Path, workspace: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "validate",
            "--input",
            str(sensor_geometry_raster),
            "--dem",
            str(workspace / "absent-dem.tif"),
        ],
    )

    assert isinstance(result.exception, MissingDEMError)
    assert "[FAIL] dem" in result.output


def test_strict_mode_turns_a_warning_into_a_failure(runner: CliRunner, workspace: Path) -> None:
    # A raster without a NoData value is a warning by default, blocking under --strict.
    raster = write_geotiff(workspace / "no_nodata.tif", nodata=None)

    lenient = runner.invoke(app, ["validate", "--input", str(raster)])
    strict = runner.invoke(app, ["validate", "--input", str(raster), "--strict"])

    assert lenient.exit_code == 0, lenient.output
    assert "[WARN] nodata_configured" in lenient.output
    assert isinstance(strict.exception, ProductValidationError)
    assert "warnings are treated as errors" in strict.output


# --------------------------------------------------------------------------------------
# Exit codes through the console-script entry point
# --------------------------------------------------------------------------------------


def test_successful_command_exits_zero(
    monkeypatch: pytest.MonkeyPatch, sample_raster: Path
) -> None:
    assert _run_main(monkeypatch, "inspect", "--input", str(sample_raster)) == 0


def test_missing_input_exits_with_the_validation_code(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    assert _run_main(monkeypatch, "inspect", "--input", str(workspace / "absent")) == 4


def test_corrupt_raster_exits_with_the_io_code(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    corrupt = workspace / "corrupt.tif"
    corrupt.write_bytes(b"nope")

    assert _run_main(monkeypatch, "inspect", "--input", str(corrupt)) == 5


def test_failed_validation_exits_with_the_validation_code(
    monkeypatch: pytest.MonkeyPatch, sample_raster: Path
) -> None:
    assert _run_main(monkeypatch, "validate", "--input", str(sample_raster), "--require-rpc") == 4


def test_error_message_and_hint_are_printed_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_raster: Path,
) -> None:
    _run_main(monkeypatch, "validate", "--input", str(sample_raster), "--require-rpc")

    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "Hint:" in stderr


def test_preview_writes_a_png(runner: CliRunner, sample_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "preview",
            "--input",
            str(sample_raster),
            "--output-dir",
            str(tmp_path),
            "--composite",
            "true-color",
            "--max-dimension",
            "32",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "scene_preview_true_color.png").is_file()
    assert "wrote" in result.stdout


def test_preview_json_reports_the_output_path(
    runner: CliRunner, sample_raster: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "preview",
            "--input",
            str(sample_raster),
            "--output-dir",
            str(tmp_path),
            "--band",
            "2",
            "--composite",
            "band",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["composite"] == "band"
    assert payload["band_indices"] == [2]
    assert Path(payload["path"]).is_file()
