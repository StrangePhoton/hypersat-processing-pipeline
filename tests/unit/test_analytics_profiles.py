"""Tests for spectral profile extraction."""

from __future__ import annotations

import numpy as np
import pytest

from hypersat.analytics.profiles import extract_profile


def test_single_pixel_profile_returns_each_band() -> None:
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    profile = extract_profile(
        data,
        row=1,
        col=0,
        band_indices=(1, 2, 3),
        wavelengths_nm=(490.0, 560.0, 665.0),
    )

    assert profile.values == (2.0, 6.0, 10.0)
    assert profile.wavelengths_nm == (490.0, 560.0, 665.0)
    assert profile.window_size == 1


def test_neighbourhood_mean_ignores_masked_samples() -> None:
    data = np.ones((1, 3, 3), dtype=np.float32)
    data[0, 1, 1] = 10.0
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 0, 0] = True

    profile = extract_profile(
        data,
        row=1,
        col=1,
        band_indices=(1,),
        mask=mask,
        window_size=3,
    )

    # Eight finite unmasked ones and one ten at the centre → mean 18/8? Wait:
    # 3x3 all ones except centre 10; one corner masked → 7 ones + one 10 = 17/8
    assert profile.values[0] == pytest.approx(17.0 / 8.0)


def test_out_of_bounds_pixel_is_rejected() -> None:
    data = np.zeros((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="outside"):
        extract_profile(data, row=2, col=0, band_indices=(1,))


def test_window_size_must_be_odd() -> None:
    data = np.zeros((1, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="odd"):
        extract_profile(data, row=1, col=1, band_indices=(1,), window_size=2)
