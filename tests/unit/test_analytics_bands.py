"""Tests for wavelength-driven band selection."""

from __future__ import annotations

import pytest

from hypersat.analytics.bands import (
    DEFAULT_TOLERANCE_NM,
    BandMatch,
    false_colour_bands,
    nearest_band,
    nearest_bands,
    true_colour_bands,
)
from hypersat.exceptions import InvalidWavelengthMetadataError

HYPERSPECTRAL = [420.0 + 10.0 * step for step in range(30)]  # 420-710 nm at 10 nm sampling


def test_an_exact_hit_reports_zero_distance() -> None:
    match = nearest_band([490.0, 560.0, 665.0], 560.0)

    assert match == BandMatch(index=2, wavelength_nm=560.0, target_nm=560.0, distance_nm=0.0)


def test_the_closest_band_wins_and_the_miss_is_reported() -> None:
    match = nearest_band(HYPERSPECTRAL, 663.0)

    assert match.index == 25
    assert match.wavelength_nm == 660.0
    assert match.distance_nm == pytest.approx(3.0)


def test_indices_are_one_based_so_they_can_be_handed_to_the_reader() -> None:
    assert nearest_band([500.0, 600.0], 500.0).index == 1


def test_a_target_the_product_does_not_cover_fails_loudly() -> None:
    # Without a tolerance this would silently return the 710 nm edge band for a SWIR
    # request, and the caller would never learn the question was unanswerable.
    with pytest.raises(InvalidWavelengthMetadataError) as excinfo:
        nearest_band(HYPERSPECTRAL, 1600.0)

    error = excinfo.value
    assert error.context["closest_band"] == 30
    assert error.context["distance_nm"] == pytest.approx(890.0)
    assert error.exit_code == 4


def test_a_wider_tolerance_accepts_a_more_distant_band() -> None:
    match = nearest_band([500.0, 600.0], 640.0, tolerance_nm=50.0)

    assert match.index == 2
    assert match.distance_nm == pytest.approx(40.0)


def test_a_tie_resolves_to_the_lower_band_index() -> None:
    # Detector overlap makes this real: EnMAP's VNIR and SWIR both observe around 950 nm.
    match = nearest_band([940.0, 960.0], 950.0, tolerance_nm=DEFAULT_TOLERANCE_NM)

    assert match.index == 1


def test_bands_without_wavelength_metadata_are_skipped() -> None:
    match = nearest_band([None, 560.0, None], 558.0)

    assert match.index == 2


def test_implausible_wavelength_values_are_ignored() -> None:
    match = nearest_band([float("nan"), -5.0, 0.0, 560.0], 560.0)

    assert match.index == 4


def test_a_product_without_any_wavelengths_cannot_be_searched() -> None:
    with pytest.raises(InvalidWavelengthMetadataError, match="No band has a usable centre"):
        nearest_band([None, None], 560.0)


def test_an_empty_band_list_cannot_be_searched() -> None:
    with pytest.raises(InvalidWavelengthMetadataError):
        nearest_band([], 560.0)


@pytest.mark.parametrize("target", [0.0, -10.0, float("nan"), float("inf")])
def test_a_nonsensical_target_is_a_programming_error(target: float) -> None:
    with pytest.raises(ValueError, match="target_nm must be"):
        nearest_band([560.0], target)


def test_a_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_nm must be"):
        nearest_band([560.0], 560.0, tolerance_nm=-1.0)


def test_several_targets_keep_the_order_they_were_asked_for() -> None:
    matches = nearest_bands(HYPERSPECTRAL, (660.0, 560.0, 490.0))

    assert [match.index for match in matches] == [25, 15, 8]


def test_two_targets_may_resolve_to_the_same_band() -> None:
    matches = nearest_bands([560.0, 665.0], (660.0, 670.0), tolerance_nm=10.0)

    assert [match.index for match in matches] == [2, 2]


def test_an_empty_target_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one wavelength"):
        nearest_bands(HYPERSPECTRAL, ())


def test_true_colour_returns_red_green_blue_in_that_order() -> None:
    red, green, blue = true_colour_bands(HYPERSPECTRAL)

    assert red.wavelength_nm > green.wavelength_nm > blue.wavelength_nm
    assert (red.index, green.index, blue.index) == (25, 15, 8)


def test_false_colour_puts_near_infrared_first() -> None:
    wavelengths = [490.0, 560.0, 665.0, 842.0]

    nir, red, green = false_colour_bands(wavelengths)

    assert (nir.index, red.index, green.index) == (4, 3, 2)


def test_false_colour_fails_on_a_visible_only_product() -> None:
    with pytest.raises(InvalidWavelengthMetadataError):
        false_colour_bands(HYPERSPECTRAL)
