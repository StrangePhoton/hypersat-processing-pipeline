"""Tests for raster inspection: metadata extraction, wavelengths, RPC and path resolution."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hypersat.exceptions import ProductStructureError, RasterReadError
from hypersat.io.inspect import (
    extract_wavelength_nm,
    inspect_input,
    inspect_raster,
    resolve_raster_path,
    scan_product_directory,
)
from hypersat.models.product import InputKind, WavelengthSource
from tests.support.rasters import SYNTHETIC_RPC_TAGS, write_enmap_like_product, write_geotiff

# --------------------------------------------------------------------------------------
# Wavelength interpretation
# --------------------------------------------------------------------------------------


def test_wavelength_from_band_tag_in_nanometres() -> None:
    value, source = extract_wavelength_nm({"wavelength": "665.4", "wavelength_units": "nm"})

    assert value == pytest.approx(665.4)
    assert source is WavelengthSource.BAND_TAG


def test_wavelength_from_band_tag_in_micrometres_is_converted() -> None:
    value, source = extract_wavelength_nm({"wavelength": "0.6654", "wavelength_units": "um"})

    assert value == pytest.approx(665.4)
    assert source is WavelengthSource.BAND_TAG


def test_wavelength_unit_is_inferred_when_absent() -> None:
    # EnMAP-like ranges do not overlap: 0.42-2.45 um vs 420-2450 nm.
    micrometre_value, _ = extract_wavelength_nm({"wavelength": "2.45"})
    nanometre_value, _ = extract_wavelength_nm({"wavelength": "2450"})

    assert micrometre_value == pytest.approx(2450.0)
    assert nanometre_value == pytest.approx(2450.0)


def test_wavelength_from_imagery_domain() -> None:
    value, source = extract_wavelength_nm({}, {"CENTRAL_WAVELENGTH_UM": "0.842"})

    assert value == pytest.approx(842.0)
    assert source is WavelengthSource.IMAGERY_DOMAIN


def test_wavelength_parsed_from_description_as_last_resort() -> None:
    value, source = extract_wavelength_nm({}, None, "Band 42 (655.4 nm)")

    assert value == pytest.approx(655.4)
    assert source is WavelengthSource.DESCRIPTION


def test_band_tag_wins_over_description() -> None:
    value, source = extract_wavelength_nm(
        {"wavelength": "700", "wavelength_units": "nm"}, None, "500 nm"
    )

    assert value == pytest.approx(700.0)
    assert source is WavelengthSource.BAND_TAG


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {"wavelength": "not-a-number"},
        {"wavelength": "0"},
        {"wavelength": "-5", "wavelength_units": "nm"},
        {"wavelength": "99999999", "wavelength_units": "nm"},
    ],
)
def test_unusable_wavelength_metadata_yields_none(tags: dict[str, str]) -> None:
    assert extract_wavelength_nm(tags) == (None, None)


# --------------------------------------------------------------------------------------
# Raster inspection
# --------------------------------------------------------------------------------------


def test_inspect_reports_core_metadata(sample_raster: Path) -> None:
    info = inspect_raster(sample_raster)

    assert info.driver == "GTiff"
    assert (info.width, info.height) == (8, 6)
    assert info.band_count == 3
    assert info.dtypes == ["uint16", "uint16", "uint16"]
    assert info.nodata == 0.0
    assert info.crs.is_defined
    assert info.crs.epsg == 32633
    assert info.crs.authority_code == "EPSG:32633"
    assert info.crs.is_projected is True
    assert info.crs.linear_units == "metre"
    assert info.has_affine_georeferencing
    assert info.pixel_size == (30.0, 30.0)
    assert info.bounds == (500000.0, 4999820.0, 500240.0, 5000000.0)
    assert info.transform[:3] == [30.0, 0.0, 500000.0]
    assert info.metadata["SENSOR"] == "synthetic"
    assert info.estimated_uncompressed_bytes == 8 * 6 * 3 * 2
    assert info.file.size_bytes > 0


def test_inspect_reports_per_band_details(sample_raster: Path) -> None:
    info = inspect_raster(sample_raster)

    descriptions = [band.description for band in info.bands]
    wavelengths = [band.wavelength_nm for band in info.bands]

    assert descriptions == ["blue", "green", "red"]
    assert wavelengths == [490.0, 560.0, 665.0]
    assert info.wavelengths_nm == [490.0, 560.0, 665.0]
    assert all(band.wavelength_source is WavelengthSource.BAND_TAG for band in info.bands)
    assert [band.index for band in info.bands] == [1, 2, 3]


def test_inspect_multi_band_raster_reports_every_band(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "cube.tif", count=24, width=4, height=4)

    info = inspect_raster(path)

    assert info.band_count == 24
    assert len(info.bands) == 24


def test_inspect_can_restrict_the_report_to_a_band_subset(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "cube.tif", count=10, width=4, height=4)

    info = inspect_raster(path, band_subset=[2, 5])

    assert info.band_count == 10
    assert [band.index for band in info.bands] == [2, 5]


def test_inspect_rejects_bands_outside_the_range(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "cube.tif", count=3, width=4, height=4)

    with pytest.raises(ProductStructureError) as raised:
        inspect_raster(path, band_subset=[1, 99])

    assert raised.value.context["out_of_range"] == [99]


def test_inspect_raster_without_crs(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "no_crs.tif", crs=None, count=1)

    info = inspect_raster(path)

    assert info.crs.is_defined is False
    assert info.crs.epsg is None
    assert info.has_affine_georeferencing is False
    assert info.transform_is_identity is True
    # Bounds from an identity transform would be pixel coordinates masquerading as map
    # coordinates, so they are deliberately not reported.
    assert info.bounds is None
    assert info.pixel_size is None
    assert any("no CRS" in warning for warning in info.warnings)


def test_inspect_raster_without_nodata(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "no_nodata.tif", nodata=None, count=2)

    info = inspect_raster(path)

    assert info.nodata is None
    assert info.nodata_is_nan is False
    assert info.nodata_per_band == [None, None]
    assert any("NoData" in warning for warning in info.warnings)


def test_inspect_reports_nan_nodata_through_a_flag(tmp_path: Path) -> None:
    # NaN cannot be represented in JSON, so it is reported as a boolean instead.
    path = write_geotiff(tmp_path / "nan_nodata.tif", dtype="float32", nodata=math.nan, count=1)

    info = inspect_raster(path)

    assert info.nodata is None
    assert info.nodata_is_nan is True


def test_inspect_reports_storage_layout(tmp_path: Path) -> None:
    path = write_geotiff(
        tmp_path / "tiled.tif", width=32, height=32, tiled=True, compress="deflate"
    )

    info = inspect_raster(path)

    assert info.is_tiled is True
    assert info.block_shapes[0] == (16, 16)
    assert info.compression == "deflate"
    assert "IMAGE_STRUCTURE" in info.metadata_domains


def test_inspect_missing_file_is_a_structure_error(tmp_path: Path) -> None:
    with pytest.raises(ProductStructureError) as raised:
        resolve_raster_path(tmp_path / "absent.tif")

    assert raised.value.exit_code == 4
    assert "data/README.md" in str(raised.value)


def test_inspect_corrupt_file_is_a_read_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"this is not a GeoTIFF")

    with pytest.raises(RasterReadError) as raised:
        inspect_raster(corrupt)

    assert raised.value.exit_code == 5
    assert raised.value.hint is not None
    assert "reason" in raised.value.context


# --------------------------------------------------------------------------------------
# RPC detection
# --------------------------------------------------------------------------------------


def test_rpc_is_detected_and_reported_as_usable(sensor_geometry_raster: Path) -> None:
    info = inspect_raster(sensor_geometry_raster)

    assert info.rpc.available is True
    assert info.rpc.is_usable is True
    assert info.rpc.issues == []
    assert info.rpc.height_off == pytest.approx(300.0)
    assert info.rpc.coefficient_counts["line_num_coeff"] == 20
    assert "RPC" in info.metadata_domains
    # A sensor-geometry product is georeferenced by its RPC model, not by an affine.
    assert info.has_affine_georeferencing is False


def test_absent_rpc_is_reported_as_unavailable(sample_raster: Path) -> None:
    info = inspect_raster(sample_raster)

    assert info.rpc.available is False
    assert info.rpc.is_usable is False


def test_partial_rpc_metadata_is_discarded_by_gdal(tmp_path: Path) -> None:
    # Verified against GDAL 3.12: an RPC domain missing required keys is dropped in full,
    # so a partially written model surfaces as "no sensor model" rather than a broken one.
    path = write_geotiff(
        tmp_path / "partial_rpc.tif",
        crs=None,
        rpc_tags={"LINE_OFF": "3.0", "SAMP_OFF": "4.0"},
    )

    info = inspect_raster(path)

    assert info.rpc.available is False


def test_degenerate_rpc_polynomial_is_reported_as_unusable(tmp_path: Path) -> None:
    # GDAL zero-pads a short coefficient list to 20 terms, which turns a truncated
    # product into an all-zero polynomial instead of a short one.
    truncated = dict(SYNTHETIC_RPC_TAGS)
    truncated["SAMP_NUM_COEFF"] = " ".join(["0.0"] * 5)

    path = write_geotiff(tmp_path / "bad_rpc.tif", crs=None, rpc_tags=truncated)
    info = inspect_raster(path)

    assert info.rpc.available is True
    assert info.rpc.is_usable is False
    assert info.rpc.coefficient_counts["samp_num_coeff"] == 20
    assert any("samp_num_coeff" in issue and "zero" in issue for issue in info.rpc.issues)


def test_zero_normalisation_scale_is_reported_as_unusable(tmp_path: Path) -> None:
    degenerate = dict(SYNTHETIC_RPC_TAGS)
    degenerate["HEIGHT_SCALE"] = "0.0"

    path = write_geotiff(tmp_path / "zero_scale.tif", crs=None, rpc_tags=degenerate)
    info = inspect_raster(path)

    assert info.rpc.available is True
    assert info.rpc.is_usable is False
    assert any("height_scale" in issue for issue in info.rpc.issues)


# --------------------------------------------------------------------------------------
# Product directories
# --------------------------------------------------------------------------------------


def test_product_directory_resolves_the_spectral_image(product_directory: Path) -> None:
    resolved, layout = resolve_raster_path(product_directory)

    assert resolved.name.endswith("-SPECTRAL_IMAGE.TIF")
    assert layout is not None
    assert layout.total_size_bytes > 0
    assert len(layout.metadata_files) == 1


def test_product_directory_prefers_spectral_image_over_quicklooks(tmp_path: Path) -> None:
    root = write_enmap_like_product(tmp_path / "product", extra_raster=True)

    resolved, layout = resolve_raster_path(root)

    assert resolved.name.endswith("-SPECTRAL_IMAGE.TIF")
    assert layout is not None
    assert len(layout.raster_candidates) == 2


def test_ambiguous_product_directory_asks_for_an_explicit_file(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    write_geotiff(root / "one.tif", count=1)
    write_geotiff(root / "two.tif", count=1)

    with pytest.raises(ProductStructureError) as raised:
        resolve_raster_path(root)

    assert "--input" in str(raised.value)
    assert len(raised.value.context["candidates"]) == 2


def test_directory_without_rasters_is_an_actionable_error(tmp_path: Path) -> None:
    root = tmp_path / "empty_product"
    root.mkdir()
    (root / "METADATA.XML").write_text("<product/>", encoding="utf-8")

    with pytest.raises(ProductStructureError) as raised:
        resolve_raster_path(root)

    assert raised.value.context["metadata_files_found"] == 1


def test_scan_product_directory_classifies_files(product_directory: Path) -> None:
    layout = scan_product_directory(product_directory)

    assert len(layout.raster_candidates) == 1
    assert len(layout.metadata_files) == 1
    assert layout.total_size_human.endswith(("B", "KiB", "MiB"))


# --------------------------------------------------------------------------------------
# Full inspection result and JSON serialisation
# --------------------------------------------------------------------------------------


def test_inspect_input_on_a_file(sample_raster: Path) -> None:
    result = inspect_input(sample_raster)

    assert result.input_kind is InputKind.FILE
    assert result.product is None
    assert result.resolved_raster_path == sample_raster
    assert result.environment.rasterio_version


def test_inspect_input_on_a_product_directory(product_directory: Path) -> None:
    result = inspect_input(product_directory)

    assert result.input_kind is InputKind.DIRECTORY
    assert result.product is not None
    assert result.raster.rpc.available is True


def test_inspection_result_serialises_to_valid_json(product_directory: Path) -> None:
    result = inspect_input(product_directory)

    payload = json.loads(result.model_dump_json())

    assert payload["input_kind"] == "directory"
    assert payload["raster"]["rpc"]["available"] is True
    assert payload["raster"]["rpc"]["is_usable"] is True
    assert isinstance(payload["raster"]["path"], str)
    assert payload["raster"]["bands"][0]["wavelength_source"] == "band_tag"


def test_json_output_never_contains_the_invalid_nan_token(tmp_path: Path) -> None:
    path = write_geotiff(tmp_path / "nan.tif", dtype="float32", nodata=math.nan, count=1)

    rendered = inspect_input(path).model_dump_json()

    assert "NaN" not in rendered
    assert json.loads(rendered)["raster"]["nodata_is_nan"] is True


def test_checksum_is_only_computed_on_request(sample_raster: Path) -> None:
    without = inspect_input(sample_raster)
    with_checksum = inspect_input(sample_raster, compute_checksum=True)

    assert without.raster.file.sha256 is None
    assert with_checksum.raster.file.sha256 is not None
    assert len(with_checksum.raster.file.sha256) == 64
