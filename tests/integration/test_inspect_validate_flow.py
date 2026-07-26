"""Integration tests for the inspect -> validate flow on a generated product directory.

Scope note: the fixture below imitates the *layout* of an EnMAP L1B product and carries a
structurally complete but scientifically meaningless RPC model. These tests therefore
verify software behaviour end to end - path resolution, metadata extraction, check
aggregation, JSON contracts, exit codes - and explicitly do **not** validate real
satellite orthorectification or geometric accuracy. Nothing produced here is a
satellite-processing result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hypersat.cli import app
from hypersat.exceptions import MissingRPCMetadataError
from tests.support.rasters import write_dem, write_enmap_like_product

pytestmark = pytest.mark.integration


def test_inspect_then_validate_a_product_directory(runner: CliRunner, tmp_path: Path) -> None:
    product = write_enmap_like_product(tmp_path / "product", band_count=6)
    dem = write_dem(tmp_path / "dem" / "dem.tif")
    output_dir = tmp_path / "outputs" / "run-1"

    inspection = runner.invoke(app, ["inspect", "--input", str(product), "--json"])
    assert inspection.exit_code == 0, inspection.output
    # Machine-readable output goes to stdout; diagnostics go to stderr, so a `--json` run
    # can be piped straight into another tool.
    raster = json.loads(inspection.stdout)["raster"]

    # A product in sensor geometry: no affine grid, but a complete sensor model and
    # per-band wavelengths - exactly the input the geometry milestone will need.
    assert raster["band_count"] == 6
    assert raster["has_affine_georeferencing"] is False
    assert raster["rpc"]["available"] is True
    assert raster["rpc"]["is_usable"] is True
    assert [band["wavelength_nm"] for band in raster["bands"]][:2] == [420.0, 520.0]

    validation = runner.invoke(
        app,
        [
            "validate",
            "--input",
            str(product),
            "--require-rpc",
            "--require-wavelengths",
            "--dem",
            str(dem),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )
    assert validation.exit_code == 0, validation.output
    report = json.loads(validation.stdout)

    assert report["is_valid"] is True
    assert report["summary"]["failed"] == 0
    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["product_structure"] == "passed"
    assert statuses["rpc_metadata"] == "passed"
    assert statuses["wavelength_metadata"] == "passed"
    assert statuses["dem"] == "passed"
    assert statuses["output_directory"] == "passed"
    assert output_dir.is_dir()


def test_validation_collects_every_problem_in_one_pass(runner: CliRunner, tmp_path: Path) -> None:
    # A product without a sensor model, an absent DEM and a size limit that cannot be met.
    product = write_enmap_like_product(tmp_path / "product", with_rpc=False)

    result = runner.invoke(
        app,
        [
            "validate",
            "--input",
            str(product),
            "--require-rpc",
            "--dem",
            str(tmp_path / "absent-dem.tif"),
            "--max-uncompressed-gb",
            "0.0000001",
            "--json",
        ],
    )

    assert isinstance(result.exception, MissingRPCMetadataError)
    report = json.loads(result.stdout)
    failed = {check["name"] for check in report["checks"] if check["status"] == "failed"}

    assert {"rpc_metadata", "dem", "product_size"} <= failed
    assert report["is_valid"] is False
    # Every failing check carries an actionable hint.
    assert all(check["hint"] for check in report["checks"] if check["status"] == "failed")


def test_orthorectification_prerequisites_fail_loudly_without_a_sensor_model(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Missing RPCs must be a hard failure, never a fallback to plain reprojection."""
    product = write_enmap_like_product(tmp_path / "product", with_rpc=False)

    result = runner.invoke(app, ["validate", "--input", str(product), "--require-rpc"])

    assert isinstance(result.exception, MissingRPCMetadataError)
    assert result.exception.exit_code == 4
    assert "not a substitute" in (result.exception.hint or "")
