"""Tests for pre-flight validation: checks, report semantics and exception mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypersat.exceptions import (
    InvalidWavelengthMetadataError,
    MissingDEMError,
    MissingGeoreferencingError,
    MissingRPCMetadataError,
    OutputPathError,
    ProductStructureError,
    ProductValidationError,
    UnreadableDEMError,
)
from hypersat.models.config import (
    InputConfig,
    OutputConfig,
    ValidationRequest,
    ValidationRequirements,
)
from hypersat.models.validation import CheckStatus, ValidationCheck, ValidationReport
from hypersat.processing.validation import (
    raise_if_invalid,
    validate_dem,
    validate_output_directory,
    validate_request,
)
from tests.support.rasters import SYNTHETIC_RPC_TAGS, write_dem, write_geotiff


def _request(path: Path, **kwargs: object) -> ValidationRequest:
    """Build a validation request for ``path`` with requirement overrides."""
    requirement_fields = set(ValidationRequirements.model_fields)
    requirements = {key: value for key, value in kwargs.items() if key in requirement_fields}
    rest = {key: value for key, value in kwargs.items() if key not in requirement_fields}
    return ValidationRequest(
        product=InputConfig(path=path),
        requirements=ValidationRequirements(**requirements),
        **rest,
    )


def _check(report: ValidationReport, name: str) -> ValidationCheck:
    """Return the single check called ``name``."""
    matches = [check for check in report.checks if check.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} check, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------------------------
# Report semantics
# --------------------------------------------------------------------------------------


def test_report_counts_and_validity() -> None:
    report = ValidationReport(
        input_path=Path("scene.tif"),
        checks=[
            ValidationCheck(name="a", status=CheckStatus.PASSED, message="ok"),
            ValidationCheck(name="b", status=CheckStatus.WARNING, message="hmm"),
            ValidationCheck(name="c", status=CheckStatus.SKIPPED, message="n/a"),
        ],
    )

    assert report.is_valid is True
    assert report.counts() == {"passed": 1, "failed": 0, "warning": 1, "skipped": 1}


def test_strict_mode_turns_warnings_into_blockers() -> None:
    checks = [ValidationCheck(name="b", status=CheckStatus.WARNING, message="hmm")]
    lenient = ValidationReport(input_path=Path("scene.tif"), checks=checks)
    strict = ValidationReport(
        input_path=Path("scene.tif"), checks=checks, treat_warnings_as_errors=True
    )

    assert lenient.is_valid is True
    assert strict.is_valid is False
    assert [check.name for check in strict.blocking] == ["b"]


def test_raise_if_invalid_is_a_no_op_for_a_valid_report() -> None:
    report = ValidationReport(
        input_path=Path("scene.tif"),
        checks=[ValidationCheck(name="a", status=CheckStatus.PASSED, message="ok")],
    )

    raise_if_invalid(report)


def test_raise_if_invalid_reraises_the_recorded_exception_type() -> None:
    report = ValidationReport(
        input_path=Path("scene.tif"),
        checks=[
            ValidationCheck(
                name="rpc_metadata",
                status=CheckStatus.FAILED,
                message="no RPC",
                hint="use L1B",
                error_type="MissingRPCMetadataError",
            ),
            ValidationCheck(name="dem", status=CheckStatus.FAILED, message="no DEM"),
        ],
    )

    with pytest.raises(MissingRPCMetadataError) as raised:
        raise_if_invalid(report)

    assert raised.value.hint == "use L1B"
    assert raised.value.context["check"] == "rpc_metadata"
    assert raised.value.context["other_blocking_checks"] == ["dem"]


def test_unknown_error_type_falls_back_to_the_base_validation_error() -> None:
    report = ValidationReport(
        input_path=Path("scene.tif"),
        checks=[
            ValidationCheck(
                name="mystery", status=CheckStatus.FAILED, message="?", error_type="Nonsense"
            )
        ],
    )

    with pytest.raises(ProductValidationError):
        raise_if_invalid(report)


# --------------------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------------------


def test_output_directory_is_created_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "run-1"

    validate_output_directory(target)

    assert target.is_dir()
    assert not any(target.iterdir()), "the write probe must be cleaned up"


def test_output_directory_rejects_a_file_path(tmp_path: Path) -> None:
    target = tmp_path / "outputs"
    target.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OutputPathError) as raised:
        validate_output_directory(target)

    assert raised.value.exit_code == 4


# --------------------------------------------------------------------------------------
# DEM
# --------------------------------------------------------------------------------------


def test_dem_validation_accepts_a_georeferenced_dem(dem_raster: Path) -> None:
    info = validate_dem(dem_raster)

    assert info.band_count == 1
    assert info.crs.epsg == 4326


def test_missing_dem_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(MissingDEMError) as raised:
        validate_dem(tmp_path / "absent-dem.tif")

    assert "docs/data-sources.md" in str(raised.value)
    assert raised.value.exit_code == 4


def test_dem_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MissingDEMError, match="not a file"):
        validate_dem(tmp_path)


def test_corrupt_dem_is_unreadable(tmp_path: Path) -> None:
    corrupt = tmp_path / "dem.tif"
    corrupt.write_bytes(b"not elevation data")

    with pytest.raises(UnreadableDEMError):
        validate_dem(corrupt)


def test_dem_without_georeferencing_is_rejected(tmp_path: Path) -> None:
    # Heights without a CRS and transform cannot be placed on the ground.
    path = write_dem(tmp_path / "plain-dem.tif", crs=None)

    with pytest.raises(UnreadableDEMError, match="cannot be located"):
        validate_dem(path)


# --------------------------------------------------------------------------------------
# Full validation pass
# --------------------------------------------------------------------------------------


def test_valid_product_passes(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster))

    assert report.is_valid
    assert _check(report, "raster_readable").status is CheckStatus.PASSED
    assert _check(report, "georeferencing").status is CheckStatus.PASSED
    assert _check(report, "proj_database").status in (CheckStatus.PASSED, CheckStatus.WARNING)


def test_missing_input_stops_after_the_structure_check(tmp_path: Path) -> None:
    report = validate_request(_request(tmp_path / "absent"))

    assert not report.is_valid
    assert [check.name for check in report.failures] == ["product_structure"]
    assert _check(report, "product_structure").error_type == "ProductStructureError"

    with pytest.raises(ProductStructureError):
        raise_if_invalid(report)


def test_corrupt_raster_stops_after_the_readability_check(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"nonsense")

    report = validate_request(_request(corrupt))

    assert not report.is_valid
    assert _check(report, "raster_readable").error_type == "RasterReadError"


def test_missing_nodata_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "no_nodata.tif", nodata=None)

    report = validate_request(_request(path))

    assert report.is_valid
    assert _check(report, "nodata_configured").status is CheckStatus.WARNING


def test_raster_without_any_georeferencing_fails(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "plain.tif", crs=None, rpc_tags=None)

    report = validate_request(_request(path))

    check = _check(report, "georeferencing")
    assert check.status is CheckStatus.FAILED
    assert check.error_type == "MissingGeoreferencingError"

    with pytest.raises(MissingGeoreferencingError):
        raise_if_invalid(report)


def test_georeferencing_requirement_can_be_relaxed(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "plain.tif", crs=None, rpc_tags=None)

    report = validate_request(_request(path, require_georeferencing=False))

    assert _check(report, "georeferencing").status is CheckStatus.WARNING
    assert report.is_valid


def test_sensor_geometry_raster_is_georeferenced_by_its_rpc_model(
    sensor_geometry_raster: Path,
) -> None:
    report = validate_request(_request(sensor_geometry_raster, require_rpc=True))

    assert report.is_valid
    assert "RPC sensor model" in _check(report, "georeferencing").message
    assert _check(report, "rpc_metadata").status is CheckStatus.PASSED


def test_required_rpc_is_a_failure_when_absent(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster, require_rpc=True))

    check = _check(report, "rpc_metadata")
    assert check.status is CheckStatus.FAILED
    assert check.error_type == "MissingRPCMetadataError"
    # The failure must state that reprojection is not an acceptable substitute.
    assert "not a substitute" in (check.hint or "")

    with pytest.raises(MissingRPCMetadataError):
        raise_if_invalid(report)


def test_absent_rpc_is_only_skipped_when_not_required(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster))

    assert _check(report, "rpc_metadata").status is CheckStatus.SKIPPED
    assert report.is_valid


def test_degenerate_rpc_fails_even_though_metadata_exists(tmp_path: Path) -> None:
    degenerate = dict(SYNTHETIC_RPC_TAGS)
    degenerate["LINE_DEN_COEFF"] = " ".join(["0.0"] * 20)

    path = write_geotiff(tmp_path / "bad_rpc.tif", crs=None, rpc_tags=degenerate)
    report = validate_request(_request(path, require_rpc=True))

    check = _check(report, "rpc_metadata")
    assert check.status is CheckStatus.FAILED
    assert check.error_type == "MissingRPCMetadataError"
    assert "not usable" in check.message

    with pytest.raises(MissingRPCMetadataError):
        raise_if_invalid(report)


def test_degenerate_rpc_is_only_a_warning_when_orthorectification_is_not_requested(
    tmp_path: Path,
) -> None:
    degenerate = dict(SYNTHETIC_RPC_TAGS)
    degenerate["LINE_DEN_COEFF"] = " ".join(["0.0"] * 20)

    path = write_geotiff(tmp_path / "bad_rpc.tif", crs=None, rpc_tags=degenerate)
    report = validate_request(_request(path))

    assert _check(report, "rpc_metadata").status is CheckStatus.WARNING
    assert report.is_valid


def test_missing_wavelengths_warn_by_default_and_fail_when_required(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "no_wavelengths.tif", wavelengths_nm=None)

    lenient = validate_request(_request(path))
    strict = validate_request(_request(path, require_wavelengths=True))

    assert _check(lenient, "wavelength_metadata").status is CheckStatus.WARNING
    assert lenient.is_valid
    assert _check(strict, "wavelength_metadata").status is CheckStatus.FAILED
    with pytest.raises(InvalidWavelengthMetadataError):
        raise_if_invalid(strict)


def test_present_wavelengths_pass(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster, require_wavelengths=True))

    check = _check(report, "wavelength_metadata")
    assert check.status is CheckStatus.PASSED
    assert check.context["first_nm"] == pytest.approx(490.0)


def test_non_monotonic_wavelengths_are_a_warning(tmp_path: Path) -> None:
    # Detector overlap makes this legitimate, so it must not be a hard failure.
    path = write_geotiff(tmp_path / "overlap.tif", count=3, wavelengths_nm=[900.0, 1000.0, 950.0])

    report = validate_request(_request(path, require_wavelengths=True))

    assert _check(report, "wavelength_metadata").status is CheckStatus.WARNING
    assert report.is_valid


def test_wavelength_override_length_must_match_band_count(sample_raster: Path) -> None:
    request = ValidationRequest(
        product=InputConfig(path=sample_raster, wavelengths_nm=(400.0, 500.0)),
        requirements=ValidationRequirements(require_wavelengths=True),
    )

    report = validate_request(request)

    check = _check(report, "wavelength_metadata")
    assert check.status is CheckStatus.FAILED
    assert check.context == {"supplied": 2, "band_count": 3}


def test_size_guard_fails_a_product_larger_than_the_limit(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster, max_uncompressed_gb=1e-9))

    check = _check(report, "product_size")
    assert check.status is CheckStatus.FAILED
    assert "band subset" in (check.hint or "")


def test_size_guard_can_be_disabled(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster, max_uncompressed_gb=None))

    assert _check(report, "product_size").status is CheckStatus.SKIPPED


def test_band_subset_outside_the_range_fails(sample_raster: Path) -> None:
    request = ValidationRequest(product=InputConfig(path=sample_raster, band_subset=(1, 99)))

    report = validate_request(request)

    check = _check(report, "band_subset")
    assert check.status is CheckStatus.FAILED
    assert check.context["out_of_range"] == [99]


def test_dem_is_skipped_when_not_supplied_but_hints_when_rpc_is_required(
    sensor_geometry_raster: Path,
) -> None:
    report = validate_request(_request(sensor_geometry_raster, require_rpc=True))

    check = _check(report, "dem")
    assert check.status is CheckStatus.SKIPPED
    assert "--dem" in (check.hint or "")


def test_dem_findings_are_reported_alongside_product_findings(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    multi_band_dem = write_dem(tmp_path / "odd_dem.tif", count=2, nodata=None)
    request = ValidationRequest(
        product=InputConfig(path=sensor_geometry_raster),
        requirements=ValidationRequirements(require_rpc=True),
        dem_path=multi_band_dem,
    )

    report = validate_request(request)

    assert _check(report, "dem").status is CheckStatus.PASSED
    assert _check(report, "dem_band_count").status is CheckStatus.WARNING
    assert _check(report, "dem_nodata").status is CheckStatus.WARNING
    assert report.is_valid


def test_missing_dem_does_not_prevent_other_checks_from_running(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    request = ValidationRequest(
        product=InputConfig(path=sensor_geometry_raster),
        requirements=ValidationRequirements(require_rpc=True),
        dem_path=tmp_path / "absent-dem.tif",
        output=OutputConfig(directory=tmp_path / "outputs"),
    )

    report = validate_request(request)

    assert not report.is_valid
    assert _check(report, "dem").error_type == "MissingDEMError"
    # The point of an aggregating pass: later checks still ran.
    assert _check(report, "rpc_metadata").status is CheckStatus.PASSED
    assert _check(report, "output_directory").status is CheckStatus.PASSED


def test_non_empty_output_directory_warns_unless_overwrite_is_enabled(
    sample_raster: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "previous.tif").write_bytes(b"")

    warned = validate_request(
        ValidationRequest(
            product=InputConfig(path=sample_raster),
            output=OutputConfig(directory=output_dir),
        )
    )
    allowed = validate_request(
        ValidationRequest(
            product=InputConfig(path=sample_raster),
            output=OutputConfig(directory=output_dir, overwrite=True),
        )
    )

    assert _check(warned, "output_directory").status is CheckStatus.WARNING
    assert _check(allowed, "output_directory").status is CheckStatus.PASSED


def test_report_serialises_to_json(sample_raster: Path) -> None:
    report = validate_request(_request(sample_raster))

    payload = report.model_dump_json()

    assert '"status":"passed"' in payload.replace(" ", "")
