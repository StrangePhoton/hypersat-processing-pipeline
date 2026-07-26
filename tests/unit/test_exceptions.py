"""Tests for the exception hierarchy and its exit-code contract."""

from __future__ import annotations

import pytest

from hypersat.exceptions import (
    ConfigurationError,
    DependencyError,
    HyperSatError,
    MissingDEMError,
    MissingRPCMetadataError,
    NotImplementedYetError,
    PipelineError,
    ProcessingError,
    ProductValidationError,
    RasterIOError,
)


def test_message_includes_context_and_hint() -> None:
    error = HyperSatError(
        "Something went wrong.",
        hint="Try harder.",
        context={"band": 42, "path": "a.tif"},
    )

    rendered = str(error)
    assert "Something went wrong." in rendered
    assert "band=42" in rendered
    assert "path='a.tif'" in rendered
    assert rendered.endswith("Hint: Try harder.")


def test_message_without_extras_is_just_the_message() -> None:
    assert str(HyperSatError("Plain failure.")) == "Plain failure."


def test_to_dict_is_report_ready() -> None:
    payload = MissingDEMError("No DEM.", context={"dem_path": "dem.tif"}).to_dict()

    assert payload["error_type"] == "MissingDEMError"
    assert payload["exit_code"] == 4
    assert payload["context"] == {"dem_path": "dem.tif"}
    assert payload["hint"] is None


def test_context_is_copied_not_aliased() -> None:
    context = {"band": 1}
    error = HyperSatError("boom", context=context)
    context["band"] = 2

    assert error.context == {"band": 1}


@pytest.mark.parametrize(
    ("exception_type", "expected_exit_code"),
    [
        (HyperSatError, 1),
        (ConfigurationError, 3),
        (ProductValidationError, 4),
        (RasterIOError, 5),
        (ProcessingError, 6),
        (PipelineError, 7),
        (DependencyError, 8),
        (NotImplementedYetError, 9),
    ],
)
def test_exit_codes_match_the_documented_map(
    exception_type: type[HyperSatError], expected_exit_code: int
) -> None:
    assert exception_type.exit_code == expected_exit_code


def test_missing_rpc_is_a_validation_error() -> None:
    # Callers must be able to catch every pre-flight problem with one except clause.
    assert issubclass(MissingRPCMetadataError, ProductValidationError)
    assert issubclass(MissingRPCMetadataError, HyperSatError)
