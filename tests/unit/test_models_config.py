"""Tests for configuration model validation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from hypersat.models.config import (
    InputConfig,
    OutputConfig,
    ProductType,
    ValidationRequest,
    ValidationRequirements,
)


def test_defaults_are_conservative() -> None:
    config = InputConfig(path=Path("data/raw/product"))

    assert config.product_type is ProductType.AUTO
    assert config.band_subset is None
    assert config.wavelengths_nm is None


def test_unknown_keys_are_rejected() -> None:
    # A mistyped configuration key must be an error, never a silently ignored setting.
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InputConfig(path=Path("scene.tif"), prodcut_type="geotiff")  # type: ignore[call-arg]


def test_models_are_immutable() -> None:
    config = OutputConfig(directory=Path("outputs"))

    with pytest.raises(ValidationError):
        config.overwrite = True  # type: ignore[misc]


def test_band_subset_is_sorted_and_deduplicated_check() -> None:
    config = InputConfig(path=Path("scene.tif"), band_subset=(42, 1, 7))

    assert config.band_subset == (1, 7, 42)


@pytest.mark.parametrize(
    ("band_subset", "message"),
    [
        ((), "must not be empty"),
        ((0, 1), "1-based"),
        ((-3,), "1-based"),
        ((2, 2, 5), "duplicate"),
    ],
)
def test_invalid_band_subsets_are_rejected(band_subset: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        InputConfig(path=Path("scene.tif"), band_subset=band_subset)


def test_wavelength_override_must_be_increasing_and_positive() -> None:
    config = InputConfig(path=Path("scene.tif"), wavelengths_nm=(420.0, 560.0, 2450.0))

    assert config.wavelengths_nm == (420.0, 560.0, 2450.0)


@pytest.mark.parametrize(
    ("wavelengths", "message"),
    [
        ((), "must not be empty"),
        ((560.0, 420.0), "strictly increasing"),
        ((560.0, 560.0), "strictly increasing"),
        ((0.0, 560.0), "finite and positive"),
        ((-1.0, 560.0), "finite and positive"),
        ((math.nan, 560.0), "finite and positive"),
        ((math.inf,), "finite and positive"),
    ],
)
def test_invalid_wavelength_overrides_are_rejected(
    wavelengths: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        InputConfig(path=Path("scene.tif"), wavelengths_nm=wavelengths)


def test_user_paths_are_expanded() -> None:
    config = InputConfig(path=Path("~/products/scene.tif"))

    assert "~" not in str(config.path)


def test_max_uncompressed_size_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        ValidationRequirements(max_uncompressed_gb=0.0)


def test_size_guard_can_be_disabled_with_none() -> None:
    assert ValidationRequirements(max_uncompressed_gb=None).max_uncompressed_gb is None


def test_validation_request_defaults() -> None:
    request = ValidationRequest(product=InputConfig(path=Path("scene.tif")))

    assert request.requirements.require_georeferencing is True
    assert request.requirements.require_rpc is False
    assert request.requirements.require_wavelengths is False
    assert request.requirements.max_uncompressed_gb == pytest.approx(16.0)
    assert request.dem_path is None
    assert request.output is None
    assert request.treat_warnings_as_errors is False
    assert request.proj_autofix is True
