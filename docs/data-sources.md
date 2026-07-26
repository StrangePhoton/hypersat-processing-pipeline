# Data sources

No data is committed to this repository, and the pipeline never downloads anything by
itself: acquisition is an explicit, documented manual step. This keeps processing runs
reproducible and avoids hidden network calls inside processing functions.

## 1. Hyperspectral imagery: EnMAP

EnMAP (Environmental Mapping and Analysis Program) is a DLR hyperspectral mission with
about 224 bands spanning roughly 420-2450 nm at approximately 30 m ground sampling.

* **Where.** DLR's EOWEB GeoPortal (<https://eoweb.dlr.de>) distributes the EnMAP
  archive. Registration is required; archived products are available free of charge for
  scientific use under the mission's licence terms.
* **What to request.** An **L1B** product: radiometrically calibrated at-sensor radiance
  in *sensor geometry*, delivered with RPC coefficients and per-band centre wavelengths.
  L1B is what makes this project meaningful — an L1C product is already orthorectified,
  so there would be nothing geometric left to demonstrate.
* **Size.** A full EnMAP tile is several gigabytes. Place it under `data/raw/`, which is
  git-ignored.
* **Alternative sensors.** Any product with RPC metadata works with the geometric stages
  (PRISMA, DESIS, or commercial VHR imagery). Wavelength-based band selection needs
  per-band centre wavelengths in the metadata.

Check what you received before processing:

```bash
hypersat inspect --input data/raw/<product_dir>
```

The report tells you whether RPC metadata and wavelengths are present, which is exactly
what the later stages require.

## 2. Digital elevation model

Orthorectification needs terrain heights. The DEM must cover the full scene footprint
with a margin, and its vertical reference should match what the RPC model assumes.

| DEM | Resolution | Vertical reference | Where |
| --- | --- | --- | --- |
| Copernicus DEM GLO-30 | 30 m | EGM2008 geoid | OpenTopography (<https://opentopography.org>), or the AWS Open Data registry |
| Copernicus DEM GLO-90 | 90 m | EGM2008 geoid | Same |
| SRTM v3 | 30 m / 90 m | EGM96 geoid | USGS EarthExplorer, OpenTopography |
| ASTER GDEM v3 | 30 m | EGM96 geoid | NASA Earthdata |

Copernicus DEM GLO-30 is the recommended default: global, recent and consistently
processed. Place the DEM under `data/dem/` and point `stages.orthorectify.dem_path` at it.

**Vertical datum caveat.** RPC models are normally defined against heights above the
WGS84 *ellipsoid*, while the DEMs above store heights above a *geoid*. The difference
(geoid undulation) reaches tens of metres and translates into a horizontal error of
roughly `undulation * tan(view angle)`. This project does not convert vertical datums; it
records the DEM used and states this limitation instead of implying the correction was
handled. GDAL can apply a shift when a suitable vertical grid is available, which is why
`RPC_DEM_APPLY_VDATUM_SHIFT` is exposed in the configuration rather than hardcoded.

## 3. Test fixtures

Automated tests never use mission data. They generate tiny GeoTIFFs in-process with
rasterio (a few pixels, synthetic values, synthetic RPC coefficients where a sensor model
is needed).

**Synthetic fixtures are test inputs only.** They are never presented as processing
results, and an integration test that runs the warping code on a synthetic RPC set
verifies *software behaviour* — that the transformer is invoked, NoData is preserved and
the grid is correct — not geometric accuracy. Real geometric accuracy can only be assessed
with a real product and independent reference data.

## 4. Directory layout

```
data/
├── README.md      committed
├── raw/           input products (git-ignored)
├── dem/           elevation models (git-ignored)
└── samples/       small hand-made samples for manual experiments (git-ignored)
outputs/           generated products (git-ignored)
```

## 5. Installing the geospatial stack

Relevant to data handling, because it determines which APIs are available:

* `rasterio` ships binary wheels that **bundle libgdal and the PROJ database**, so
  `pip install rasterio` works on Linux, macOS and Windows with no system GDAL. This is
  the guaranteed dependency of the project.
* The `osgeo.gdal` Python bindings are published on PyPI as a **source distribution only**
  (verified: `gdal` 3.13.2 offers just `gdal-3.13.2.tar.gz`). Installing them requires a
  matching system libgdal and a C++ toolchain, which is why they are an optional extra
  (`pip install -e ".[gdal]"`) and why the Docker image is based on `python:slim` plus
  rasterio wheels rather than on an `osgeo/gdal` image. On Windows, prefer conda-forge or
  the container if you need the bindings.
