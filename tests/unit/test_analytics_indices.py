"""Tests for normalised-difference spectral indices."""

from __future__ import annotations

import numpy as np
import pytest

from hypersat.analytics.indices import ndvi, ndwi, normalised_difference


def test_normalised_difference_matches_the_classic_formula() -> None:
    high = np.array([[0.8, 0.4]], dtype=np.float32)
    low = np.array([[0.2, 0.4]], dtype=np.float32)

    result = normalised_difference(high, low, nodata=-9999.0)

    assert result[0, 0] == pytest.approx(0.6)
    assert result[0, 1] == pytest.approx(0.0)


def test_division_by_zero_becomes_nodata() -> None:
    high = np.array([[1.0, -2.0]], dtype=np.float32)
    low = np.array([[-1.0, 2.0]], dtype=np.float32)

    result = normalised_difference(high, low, nodata=-9999.0)

    assert result[0, 0] == pytest.approx(-9999.0)
    assert result[0, 1] == pytest.approx(-9999.0)


def test_masked_samples_become_nodata() -> None:
    high = np.array([[0.8]], dtype=np.float32)
    low = np.array([[0.2]], dtype=np.float32)
    mask = np.array([[True]])

    result = ndvi(high, low, mask=mask, nodata=-9999.0)

    assert result[0, 0] == pytest.approx(-9999.0)


def test_non_finite_samples_become_nodata() -> None:
    high = np.array([[np.nan, 0.8]], dtype=np.float32)
    low = np.array([[0.2, np.inf]], dtype=np.float32)

    result = ndwi(high, low, nodata=-9999.0)

    assert result[0, 0] == pytest.approx(-9999.0)
    assert result[0, 1] == pytest.approx(-9999.0)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        normalised_difference(np.ones((2, 2)), np.ones((3, 3)))
