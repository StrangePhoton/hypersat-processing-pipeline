"""Tests for spectral index and profile orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hypersat.exceptions import SpectralAnalysisError
from hypersat.io.inspect import inspect_raster
from hypersat.io.reader import read_chunk
from hypersat.models.config import (
    IndexRequest,
    OutputConfig,
    SpectralIndexName,
    SpectralProfileRequest,
)
from hypersat.processing.spectral import calculate_index, extract_spectral_profile


def test_ndvi_selects_nir_and_red_by_wavelength(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    result = calculate_index(
        IndexRequest(
            product_path=sensor_geometry_raster,
            output=OutputConfig(directory=tmp_path),
            index=SpectralIndexName.NDVI,
        )
    )

    assert result.band_indices == (4, 3)
    assert result.wavelengths_nm == (842.0, 665.0)
    assert result.path.name.endswith("_ndvi.tif")
    info = inspect_raster(result.path)
    assert info.band_count == 1
    assert info.nodata == pytest.approx(-9999.0)


def test_ndwi_writes_mcfeeters_index(sensor_geometry_raster: Path, tmp_path: Path) -> None:
    result = calculate_index(
        IndexRequest(
            product_path=sensor_geometry_raster,
            output=OutputConfig(directory=tmp_path),
            index=SpectralIndexName.NDWI,
        )
    )

    assert result.band_indices == (2, 4)
    chunk = read_chunk(result.path)
    assert chunk.data.dtype == np.float32


def test_index_statistics_sidecar_is_optional(sensor_geometry_raster: Path, tmp_path: Path) -> None:
    result = calculate_index(
        IndexRequest(
            product_path=sensor_geometry_raster,
            output=OutputConfig(directory=tmp_path),
            index=SpectralIndexName.NDVI,
            include_statistics=True,
        )
    )

    assert result.statistics is not None
    assert result.statistics_path is not None
    payload = json.loads(result.statistics_path.read_text(encoding="utf-8"))
    assert payload["index"] == "ndvi"
    assert payload["statistics"]["valid_count"] > 0


def test_explicit_bands_override_wavelength_selection(
    sensor_geometry_raster: Path, tmp_path: Path
) -> None:
    result = calculate_index(
        IndexRequest(
            product_path=sensor_geometry_raster,
            output=OutputConfig(directory=tmp_path),
            index=SpectralIndexName.NDVI,
            bands=(4, 2),
        )
    )

    assert result.band_indices == (4, 2)


def test_spectral_profile_writes_csv_and_json(sample_raster: Path, tmp_path: Path) -> None:
    result = extract_spectral_profile(
        SpectralProfileRequest(
            product_path=sample_raster,
            output=OutputConfig(directory=tmp_path),
            row=2,
            col=3,
        )
    )

    assert result.csv_path.is_file()
    assert result.json_path.is_file()
    assert result.profile.row == 2
    assert result.profile.col == 3
    assert len(result.profile.values) == 3
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["values"] == list(result.profile.values)


def test_spectral_profile_rejects_out_of_bounds_pixel(sample_raster: Path, tmp_path: Path) -> None:
    with pytest.raises(SpectralAnalysisError, match="outside"):
        extract_spectral_profile(
            SpectralProfileRequest(
                product_path=sample_raster,
                output=OutputConfig(directory=tmp_path),
                row=100,
                col=0,
            )
        )
