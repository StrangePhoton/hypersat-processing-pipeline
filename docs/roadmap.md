# Implementation roadmap

Small, reviewable milestones. Each one ends with green lint, type checks and tests before
the next one starts. Status is kept honest: the README's capability table mirrors it.

| # | Milestone | Scope | Status |
| --- | --- | --- | --- |
| 1 | Foundation | Repository skeleton, packaging, Ruff/mypy/pytest/pre-commit/CI, exception hierarchy, structured logging, minimal CLI (`version`, placeholder `inspect`), design docs, configuration schema, contracts | Done |
| 2 | Inspection and validation | `RasterInfo`/`BandInfo`/`RPCInfo` models, rasterio-backed metadata inspection, file size and checksum utilities, product/DEM/output-path validation, working `hypersat inspect` and `hypersat validate` with text and JSON output, GeoTIFF test fixtures | Done |
| 3 | Raster reading and band selection | Windowed, band-selective, masked reading; float32 conversion; memory guard; atomic tiled/compressed GeoTIFF writer; wavelength-to-band selection as pure functions | Planned |
| 4 | Geometry | Reprojection and grid alignment, resampling selection by data semantics, automatic UTM choice, then RPC + DEM orthorectification with explicit failure when the sensor model or DEM is unusable; `hypersat orthorectify` | Planned |
| 5 | Quality mask and previews | Class raster per `docs/quality-masks.md`, configurable OpenCV morphology, percentile stretching, RGB/false-colour/single-band PNG previews; `hypersat preview` | Planned |
| 6 | Spectral analytics | NDVI, NDWI with safe division and NoData propagation, pixel spectral profiles, per-band statistics; `hypersat calculate-index`, `hypersat spectral-profile` | Planned |
| 7 | Orchestration and QC report | `Stage` protocol, sequential runner with timing and failure isolation, YAML-driven `hypersat process`, JSON quality-control report with checksums; end-to-end integration test | Planned |
| 8 | Polish | Example outputs in the README, an `external`-marked real-product test, performance notes | Planned |

## Decisions to confirm before the milestones that need them

1. **Warp backend (needed by milestone 4).** `osgeo.gdal.Warp` is the canonical API but
   installs only from source on PyPI, so it cannot be `pip install`ed on Windows;
   `rasterio.warp.reproject(..., rpcs=..., RPC_DEM=...)` reaches the same GDAL code
   through wheels that work everywhere. Current default: rasterio as the required path,
   `osgeo.gdal` as an optional extra. Verified against the installed rasterio 1.5.0
   (GDAL 3.12.1): `rasterio.rpc.RPC` exposes the full coefficient set with
   `from_gdal()`/`to_gdal()`, and `rasterio.warp.reproject` accepts `rpcs=` plus a
   `**kwargs` passthrough for GDAL transformer options such as `RPC_DEM`. Since that is a
   binding over the same `GDALCreateRPCTransformer`/`GDALWarp` code, it is real
   orthorectification and not a substitute, so the default stands.
2. **Real product for the showcase (needed by milestone 8).** Which EnMAP scene and DEM
   tile will be used for the README's example outputs.
3. ~~**Stale PROJ data path on developer machines.**~~ **Resolved in milestone 2.** A
   machine-level `PROJ_LIB`/`PROJ_DATA` pointing at an unrelated PROJ installation — the
   PostgreSQL/PostGIS installer sets one — shadows the `proj.db` bundled in the rasterio
   wheel, and every `CRS.from_epsg()` call then fails with a
   `DATABASE.LAYOUT.VERSION.MINOR` mismatch. Verified on the development machine:
   re-pointing the environment variables after import does **not** help, because PROJ
   resolves its search path when the library loads, but
   `rasterio.env.set_proj_data_search_path()` does, because it updates that search path in
   place. `hypersat.io.environment.ensure_usable_proj_data()` therefore probes EPSG:4326
   and, only when the probe fails, falls back to the bundled database. The fallback is
   never silent: it logs a warning with the conflicting path, appears as a `proj_database`
   warning in `hypersat validate`, and can be disabled with `--no-proj-autofix`.

## Explicitly out of scope for the MVP

Machine learning, distributed execution, a web API, cloud object storage, mosaicking,
time-series analysis, cloud masking, atmospheric correction and GCP-based geometric
refinement. Rationale in `docs/limitations.md`.
