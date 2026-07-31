"""Tests for percentile stretching."""

from __future__ import annotations

import numpy as np
import pytest

from hypersat.visualization.stretch import percentile_limits, stretch_to_uint8


def test_percentile_limits_ignore_non_finite_samples() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.inf], dtype=np.float64)

    low, high = percentile_limits(samples, lower=0.0, upper=100.0)

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(5.0)


def test_percentile_limits_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="no finite samples"):
        percentile_limits(np.array([np.nan, np.inf]))


def test_stretch_maps_valid_range_onto_uint8() -> None:
    data = np.arange(100, dtype=np.float32).reshape(1, 10, 10)

    result = stretch_to_uint8(data, lower_percentile=0.0, upper_percentile=100.0)

    assert result.dtype == np.uint8
    assert result.shape == (1, 10, 10)
    assert int(result.min()) == 0
    assert int(result.max()) == 255


def test_stretch_excludes_masked_samples_from_percentiles_and_paints_them_black() -> None:
    data = np.array([[[0.0, 10.0], [20.0, 30.0]]], dtype=np.float32)
    mask = np.array([[[True, False], [False, False]]])

    result = stretch_to_uint8(
        data, mask=mask, lower_percentile=0.0, upper_percentile=100.0, per_band=True
    )

    assert int(result[0, 0, 0]) == 0
    assert int(result[0, 0, 1]) == 0
    assert int(result[0, 1, 1]) == 255


def test_joint_stretch_uses_one_shared_range() -> None:
    data = np.array(
        [
            [[0.0, 10.0], [20.0, 30.0]],
            [[100.0, 110.0], [120.0, 130.0]],
        ],
        dtype=np.float32,
    )

    result = stretch_to_uint8(data, lower_percentile=0.0, upper_percentile=100.0, per_band=False)

    # Shared limits are 0..130, so the bright band sits near the top of the range.
    assert int(result[0].max()) < int(result[1].min())


def test_constant_band_does_not_raise() -> None:
    data = np.full((1, 4, 4), 7.0, dtype=np.float32)

    result = stretch_to_uint8(data, lower_percentile=2.0, upper_percentile=98.0)

    assert result.shape == (1, 4, 4)
    assert set(result.ravel().tolist()) == {0}
