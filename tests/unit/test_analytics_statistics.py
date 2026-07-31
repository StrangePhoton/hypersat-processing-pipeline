"""Tests for per-band descriptive statistics."""

from __future__ import annotations

import numpy as np
import pytest

from hypersat.analytics.statistics import band_statistics, summarise_array


def test_summarise_array_reports_basic_moments() -> None:
    values = np.arange(1, 11, dtype=np.float32)

    stats = summarise_array(values, band_index=3)

    assert stats.band_index == 3
    assert stats.valid_count == 10
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(10.0)
    assert stats.mean == pytest.approx(5.5)


def test_nodata_and_mask_are_excluded() -> None:
    values = np.array([[1.0, -9999.0], [3.0, 5.0]], dtype=np.float32)
    mask = np.array([[False, False], [True, False]])

    stats = summarise_array(values, mask=mask, nodata=-9999.0)

    assert stats.valid_count == 2
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(5.0)


def test_empty_valid_set_returns_nones() -> None:
    values = np.array([np.nan, np.inf], dtype=np.float32)

    stats = summarise_array(values)

    assert stats.valid_count == 0
    assert stats.mean is None


def test_band_statistics_covers_every_band() -> None:
    data = np.stack(
        [
            np.full((4, 4), 1.0, dtype=np.float32),
            np.full((4, 4), 2.0, dtype=np.float32),
        ]
    )

    stats = band_statistics(data, sample_step=2)

    assert len(stats) == 2
    assert stats[0].mean == pytest.approx(1.0)
    assert stats[1].mean == pytest.approx(2.0)
    assert stats[0].valid_count == 4  # 4x4 with step 2 → 2x2
