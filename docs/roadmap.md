# Implementation roadmap

Small, reviewable milestones. Each one ends with green lint, type checks and tests before
the next one starts. Status is kept honest: the README's capability table mirrors it.

| # | Milestone | Scope | Status |
| --- | --- | --- | --- |
| 1 | Skeleton and architecture | Repository skeleton, packaging, Ruff/mypy/pytest/pre-commit/CI, Docker image and container smoke test, exception hierarchy, structured logging, minimal CLI (`version`, placeholder `inspect`), design docs, configuration schema, contracts | Done |
| 2 | Raster inspection and validation | `RasterInfo`/`BandInfo`/`RPCInfo` models, rasterio-backed metadata inspection, file size and checksum utilities, product/DEM/output-path validation, working `hypersat inspect` and `hypersat validate` with text and JSON output, GeoTIFF test fixtures | Done |
| 3 | Raster reader, windowed processing and metadata propagation | Windowed, band-selective, masked reading; float32 conversion; memory guard; atomic tiled/compressed GeoTIFF writer that carries CRS, transform, NoData and band metadata through; wavelength-to-nearest-band selection as pure functions | Done |
| 4 | Previews and OpenCV preprocessing | Percentile stretching, RGB and false-colour composites, single-band previews, configurable OpenCV resize/blur with explicit kernel sizes; `hypersat preview` | Done |
| 5 | Spectral analytics | NDVI and NDWI with safe division and NoData propagation, pixel spectral profiles, per-band descriptive statistics; `hypersat calculate-index`, `hypersat spectral-profile` | Done |
| 6 | Quality mask | Class raster per `docs/quality-masks.md`, configurable OpenCV morphology, built in sensor geometry; `hypersat quality-mask` | Done |
| 7 | Reprojection and raster alignment | CRS reprojection, output-transform calculation, resolution control, alignment to a reference grid, bounds validation, resampling choice by data semantics, automatic UTM selection; `hypersat reproject` | Done |
| 8 | RPC + DEM orthorectification | RPC sensor model plus DEM warp through GDAL (via rasterio), explicit failure when the sensor model or the DEM is unusable, DEM overlap check, logged transformer configuration; `hypersat orthorectify` | Done |
| 9 | Pipeline orchestration and YAML config | `Stage` protocol, sequential runner with timing and failure isolation, YAML-driven `hypersat process` | Planned |
| 10 | QC report and showcase | JSON quality-control report with per-stage timings and output checksums, end-to-end integration test, README example outputs from a real EnMAP scene, `external`-marked real-product test | Planned |

Two notes on the numbering. The Docker image, the Makefile and CI landed in milestone 1 and
are already verified by a container smoke test on every push, so milestone 10 is the QC
report and the showcase rather than containerisation. And wavelength-to-nearest-band
selection sits in milestone 3 rather than with the spectral analytics, because the previews
in milestone 4 have to choose their composite bands by wavelength; the alternative would be
hardcoding EnMAP band indices, which this project explicitly avoids.

## Decisions to confirm before the milestones that need them

1. ~~**Warp backend (needed by milestone 8).**~~ **Resolved in milestone 8.** The
   implementation uses `rasterio.warp.reproject(..., rpcs=..., RPC_DEM=...)`, which
   reaches the same GDAL `GDALCreateRPCTransformer` / warp code as `osgeo.gdal.Warp`
   through wheels that install on Windows. `osgeo.gdal` remains an optional extra for
   environments that already have a system libgdal.
2. **Real product for the showcase (needed by milestone 10).** Which EnMAP scene and DEM
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
