"""Unit tests for quality-mask classification and morphology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from hypersat.models.config import (
    MorphologyConfig,
    MorphologyKernelShape,
    MorphologyOperation,
    OutputConfig,
    QualityMaskRequest,
)
from hypersat.processing.quality_mask import (
    QualityClass,
    apply_defect_morphology,
    build_quality_mask,
    class_counts,
    classify_quality,
)
from tests.support.rasters import write_geotiff


def test_classify_assigns_classes_with_documented_precedence() -> None:
    # Four evaluation bands; band_fraction 0.5 means two bands are enough.
    data = np.full((4, 3, 3), 100.0, dtype=np.float64)
    data[:, 0, 0] = 0.0  # NoData
    data[0, 0, 1] = np.nan  # one NaN among finite bands -> INVALID_NUMERIC
    data[:, 0, 2] = 65535.0  # saturated
    data[:, 1, 0] = 5.0  # low signal
    data[0, 1, 1] = 65535.0  # only one band saturated -> still VALID at fraction 0.5
    data[0, 1, 2] = 65535.0
    data[1, 1, 2] = 65535.0  # two bands saturated -> SATURATED

    mask = np.zeros(data.shape, dtype=bool)
    mask[:, 0, 0] = True

    result = classify_quality(
        data,
        mask=mask,
        saturation_dn=65535.0,
        low_signal_dn=10.0,
        band_fraction=0.5,
    )

    assert result[0, 0] == QualityClass.NO_DATA
    assert result[0, 1] == QualityClass.INVALID_NUMERIC
    assert result[0, 2] == QualityClass.SATURATED
    assert result[1, 0] == QualityClass.LOW_SIGNAL
    assert result[1, 1] == QualityClass.VALID
    assert result[1, 2] == QualityClass.SATURATED


def test_invalid_numeric_beats_saturation() -> None:
    # Mix of NaN and finite samples -> INVALID_NUMERIC; pure saturation stays SATURATED.
    data = np.array(
        [
            [[np.nan, 65535.0], [65535.0, 65535.0]],
            [[100.0, 65535.0], [65535.0, 65535.0]],
        ],
        dtype=np.float64,
    )

    result = classify_quality(
        data,
        saturation_dn=65535.0,
        low_signal_dn=10.0,
        band_fraction=1.0,
    )

    assert result[0, 0] == QualityClass.INVALID_NUMERIC
    assert result[0, 1] == QualityClass.SATURATED


def test_all_nan_pixel_is_no_data() -> None:
    data = np.full((2, 1, 1), np.nan, dtype=np.float64)

    result = classify_quality(
        data,
        saturation_dn=65535.0,
        low_signal_dn=10.0,
        band_fraction=0.5,
    )

    assert result[0, 0] == QualityClass.NO_DATA


def test_spectral_anomaly_flags_high_cv_when_enabled() -> None:
    data = np.zeros((3, 1, 2), dtype=np.float64)
    data[:, 0, 0] = [10.0, 11.0, 10.5]  # low CV
    data[:, 0, 1] = [1.0, 100.0, 50.0]  # high CV

    disabled = classify_quality(
        data,
        saturation_dn=65535.0,
        low_signal_dn=0.0,
        band_fraction=1.0,
        spectral_anomaly=False,
    )
    enabled = classify_quality(
        data,
        saturation_dn=65535.0,
        low_signal_dn=0.0,
        band_fraction=1.0,
        spectral_anomaly=True,
        anomaly_cv_threshold=0.5,
    )

    assert disabled[0, 1] == QualityClass.VALID
    assert enabled[0, 0] == QualityClass.VALID
    assert enabled[0, 1] == QualityClass.SPECTRAL_ANOMALY


def test_morphology_dilate_grows_defects_over_valid_only() -> None:
    mask = np.full((5, 5), QualityClass.VALID, dtype=np.uint8)
    mask[2, 2] = QualityClass.SATURATED
    mask[0, 0] = QualityClass.NO_DATA

    grown = apply_defect_morphology(
        mask,
        MorphologyConfig(
            enabled=True,
            operation=MorphologyOperation.DILATE,
            kernel_shape=MorphologyKernelShape.RECT,
            kernel_size=3,
            iterations=1,
        ),
    )

    assert grown[2, 2] == QualityClass.SATURATED
    assert grown[2, 3] == QualityClass.SATURATED
    assert grown[1, 2] == QualityClass.SATURATED
    assert grown[0, 0] == QualityClass.NO_DATA
    assert grown[0, 1] == QualityClass.VALID  # NO_DATA is not a defect class for morphology
    assert grown[4, 4] == QualityClass.VALID


def test_morphology_never_reclassifies_to_valid() -> None:
    mask = np.full((3, 3), QualityClass.SATURATED, dtype=np.uint8)
    mask[1, 1] = QualityClass.VALID

    eroded = apply_defect_morphology(
        mask,
        MorphologyConfig(
            enabled=True,
            operation=MorphologyOperation.ERODE,
            kernel_shape=MorphologyKernelShape.RECT,
            kernel_size=3,
            iterations=1,
        ),
    )

    # Erosion of the defect binary cannot paint VALID over existing defects.
    assert eroded[0, 0] == QualityClass.SATURATED
    assert np.count_nonzero(eroded == QualityClass.VALID) <= 1


def test_disabled_morphology_is_a_noop() -> None:
    mask = np.array([[QualityClass.SATURATED, QualityClass.VALID]], dtype=np.uint8)

    assert np.array_equal(
        apply_defect_morphology(mask, MorphologyConfig(enabled=False)),
        mask,
    )


def test_class_counts_cover_every_named_class() -> None:
    mask = np.array(
        [
            [QualityClass.NO_DATA, QualityClass.VALID],
            [QualityClass.SATURATED, QualityClass.LOW_SIGNAL],
        ],
        dtype=np.uint8,
    )

    counts = class_counts(mask)

    assert counts["no_data"] == 1
    assert counts["valid"] == 1
    assert counts["saturated"] == 1
    assert counts["low_signal"] == 1
    assert counts["invalid_numeric"] == 0


def test_build_quality_mask_writes_uint8_geotiff(tmp_path: Path) -> None:
    product = write_geotiff(
        tmp_path / "scene.tif",
        width=4,
        height=4,
        count=4,
        dtype="uint16",
        crs=None,
        nodata=0.0,
        fill_value=100,
        wavelengths_nm=(490.0, 560.0, 665.0, 842.0),
        nodata_pixels=((0, 0),),
    )
    with rasterio.open(product, "r+") as dataset:
        for band_index in range(1, 5):
            values = np.array(dataset.read(band_index), copy=True)
            values[1, 1] = 65535
            values[2, 2] = 5
            dataset.write(values, band_index)

    result = build_quality_mask(
        QualityMaskRequest(
            product_path=product,
            output=OutputConfig(directory=tmp_path / "out", overwrite=False),
            bands=(1, 2, 3, 4),
            include_statistics=True,
        )
    )

    assert result.path.exists()
    assert result.statistics_path is not None
    assert result.statistics_path.exists()
    with rasterio.open(result.path) as dataset:
        assert dataset.count == 1
        assert dataset.dtypes[0] == "uint8"
        assert dataset.nodata == 0
        classes = dataset.read(1)
    assert classes[0, 0] == QualityClass.NO_DATA
    assert classes[1, 1] == QualityClass.SATURATED
    assert classes[2, 2] == QualityClass.LOW_SIGNAL
    assert result.counts["saturated"] >= 1
