# Scientific and engineering limitations

This document exists because the most valuable property of a processing pipeline is
knowing what it does *not* guarantee. Nothing here is an apology; each item is a
deliberate scope boundary.

## 1. Radiometry

* **No radiometric calibration.** Dark-current subtraction, gain/offset application,
  non-linearity, stray light and detector-response correction are not implemented. The
  pipeline consumes an already radiometrically calibrated L1B product and never claims to
  produce or improve calibration.
* **No spectral-smile or keystone correction.** Hyperspectral pushbroom instruments show
  an across-track shift of band centre wavelengths (smile) and a spectral-spatial
  misregistration (keystone). Correcting them requires the instrument's characterisation
  data. The pipeline uses the single nominal centre wavelength per band from the metadata,
  which is an approximation at the swath edges.
* **No bad-pixel or dead-column interpolation**, since that needs the instrument's
  bad-pixel table.
* **No atmospheric correction.** No aerosol or water-vapour retrieval, no adjacency
  correction, no BRDF or topographic illumination correction. Outputs are at-sensor
  radiance (or DN), never surface reflectance.

Consequence: spectral indices computed here are arithmetic on radiance. They are not
comparable across scenes, dates or sensors, and they must not be read as calibrated
biophysical measurements.

## 2. Geometry

* **Accuracy is bounded by the delivered RPC model.** No ground control points, no tie
  points, no bundle adjustment, no co-registration against a reference image. The output
  cannot be more accurate than the RPCs that came with the product.
* **No vertical datum conversion.** DEM heights are used as delivered. A geoid/ellipsoid
  mismatch produces a systematic horizontal shift (see `docs/data-sources.md`).
* **No DEM void filling or quality assessment.** Where the DEM has no data, the configured
  `RPC_DEM_MISSING_VALUE` applies, and the affected area is not flagged as lower quality.
* **No geometric accuracy validation.** The QC report describes what was done — CRS,
  grid, resampling, transformer options — and deliberately reports no CE90/RMSE figure,
  because that requires independent reference data.
* **Resampling changes pixel values.** Bilinear and cubic warping mixes neighbouring
  detector samples, which alters spectra. This is why the quality mask is computed before
  warping and always resampled with nearest-neighbour.

## 3. Masking

* **No cloud, cirrus, shadow, snow or water detection.** The quality mask covers
  NoData, saturation, low signal and numerically invalid samples only. Details and the
  reasoning are in `docs/quality-masks.md`.
* **Thresholds are configuration, not mission specifications.** The defaults shipped in
  the example configuration are illustrative.

## 4. Product level

* Output is **never** an L1C or L2A product. It is an "orthorectified demonstration
  product" or "analysis-ready demonstration output". The rules are in
  `docs/processing-levels.md`.
* The project does not recreate EnMAP's operational ground segment, does not implement raw
  telemetry decoding or L0/L1A framing, and does not reproduce mission-internal geometric
  or radiometric calibration.

## 5. Software scope

* **Sequential, single-machine execution.** No distributed processing, no job queue, no
  scheduler. Adding Celery or Airflow would demonstrate infrastructure the geospatial
  problem here does not need.
* **No web API, database or frontend.**
* **Local filesystem only.** Cloud object storage is not integrated; GDAL's `/vsis3/` and
  friends would work at the I/O boundary but are untested here.
* **One product per run.** No mosaicking, no time series, no multi-scene stacking beyond
  the optional grid alignment.
* **Memory-aware, not memory-guaranteed.** Reads are windowed and band-selective, and a
  configurable size guard exists, but a warp of a full 224-band cube is still a heavy
  operation whose peak usage depends on GDAL internals.
* **No machine learning.** PyTorch is intentionally excluded until the deterministic
  geospatial chain is complete and tested.

## 6. Testing

* Automated tests run on **synthetic fixtures** generated in-process. They verify software
  behaviour: contracts, error paths, NoData preservation, grid arithmetic, class
  assignment.
* A passing test suite therefore proves the *code* behaves as specified. It does not
  prove the *scientific* correctness of a product derived from real mission data.
* Tests that need real products are marked `external` and skipped by default.
