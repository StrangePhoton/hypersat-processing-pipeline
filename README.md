# HyperSat Processing Pipeline

A production-styled Python pipeline that inspects, validates, orthorectifies and analyses
publicly available **hyperspectral** satellite imagery — primarily EnMAP L1B products.

The goal is to demonstrate the software engineering and geospatial reasoning behind
turning a raw-ish satellite product into a reliable, analysis-ready raster product:
sensor models, digital elevation models, coordinate reference systems, resampling
semantics, quality masking and machine-readable quality control.

> **Portfolio disclaimer.** This project demonstrates *selected concepts* from an
> Earth-observation processing chain. It does **not** recreate EnMAP's official processing
> system, does **not** implement mission-specific raw telemetry decoding, and does **not**
> perform radiometric or atmospheric calibration. Products it generates are labelled
> "orthorectified demonstration product" / "analysis-ready demonstration output" and must
> never be called official L1C or L2A products. See
> [docs/limitations.md](docs/limitations.md).

## Status

The project is built in small milestones ([docs/roadmap.md](docs/roadmap.md)). This table
reflects what actually runs today — planned features are marked as such, not advertised.

| Capability | Command | Status |
| --- | --- | --- |
| Version / environment info | `hypersat version` | Working |
| Product and raster inspection | `hypersat inspect` | Working |
| Pre-flight validation (product, DEM, output path) | `hypersat validate` | Working |
| Windowed / band-selective raster reading | library API | Working |
| Atomic tiled/compressed GeoTIFF writing | library API | Working |
| Wavelength-based band selection | library API | Working |
| Percentile-stretched PNG previews | `hypersat preview` | Working |
| NDVI / NDWI, spectral profiles, band statistics | `hypersat calculate-index`, `hypersat spectral-profile` | Planned (milestone 5) |
| Quality mask | `hypersat process` stage | Planned (milestone 6) |
| Reprojection and grid alignment | library API | Planned (milestone 7) |
| RPC + DEM orthorectification | `hypersat orthorectify` | Planned (milestone 8) |
| YAML-driven pipeline | `hypersat process` | Planned (milestone 9) |
| JSON QC report | `hypersat process` | Planned (milestone 10) |

Unimplemented functionality never silently succeeds, and planned commands are absent from
the CLI rather than present and doing nothing. Exit code 9 (`NotImplementedYetError`) is
reserved for the case where a command has to refuse work it cannot honestly perform; no
command uses it at the moment.
Every other planned command is absent from `--help` until it does something real.

## Architecture

```
                        +-------------------------------+
                        |          hypersat.cli         |  Typer commands, thin
                        |  parse args -> call service   |  handlers, exit codes
                        +---------------+---------------+
                                        |
                        +---------------v---------------+
                        |       hypersat.pipeline       |  stage protocol, runner,
                        |  order, timing, QC report     |  artifacts   (milestone 9)
                        +----+--------------+-----------+
                             |              |
          +------------------v---+     +----v-----------------------+
          | hypersat.processing  |     |     hypersat.analytics     |  pure NumPy
          | validation, mask,    |     |  band selection, indices,  |  functions
          | ortho, reprojection  |     |  profiles, statistics      |
          +----------+-----------+     +----+-----------------------+
                     |                      |
                     |                 +----v-----------------------+
                     |                 |  hypersat.visualization    |  8-bit PNG only
                     |                 |  stretching, composites    |
                     |                 +----+-----------------------+
                     |                      |
          +----------v----------------------v---------------------+
          |                     hypersat.io                       |  the ONLY layer
          |  inspect, windowed reader, atomic writer, product      |  that opens files
          +----------+--------------------------------------------+
                     |
          +----------v--------------------------------------------+
          |        rasterio / GDAL / optional osgeo.gdal           |
          +-------------------------------------------------------+

          hypersat.models         pure Pydantic data, no I/O
          hypersat.exceptions     error hierarchy + exit-code contract
          hypersat.logging_config structured text/JSON logging
```

Details, module map, stage contracts and the output naming convention:
[docs/architecture.md](docs/architecture.md).

## Processing flow

```
  EnMAP L1B product (or any GeoTIFF)
            |
   [1] validate         metadata only, no pixel reads
            v
   [2] quality_mask     built in SENSOR geometry, on original detector samples
            v
   [3] orthorectify     RPC sensor model + DEM, GDAL warp
            |           imagery: nearest | bilinear | cubic; mask: nearest ONLY
            v
   [4] align            optional: reproject / snap onto a reference grid
            v
   [5] spectral_indices bands chosen by wavelength, never by band number
            v
   [6] preview          percentile-stretched 8-bit PNG, cosmetic only
            v
   [7] report           qc_report.json, always written, even after a failure
```

The quality mask is produced **before** warping because saturation and fill values are
properties of individual detector samples; after resampling, an output pixel is a mixture
of several samples and the question is no longer answerable.

## Installation

Requires Python 3.11 or newer. No system GDAL installation is needed: the `rasterio`
wheels bundle their own libgdal and PROJ database.

```bash
git clone https://github.com/StrangePhoton/hypersat-processing-pipeline.git
cd hypersat-processing-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

Or, with GNU make: `make install`.

### Troubleshooting: `DATABASE.LAYOUT.VERSION.MINOR` errors

If CRS lookups fail with `proj_create_from_database: ... contains
DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a number >= 6 is expected`, a machine-level
`PROJ_LIB` or `GDAL_DATA` variable is pointing at another PROJ installation and shadowing
the `proj.db` inside the rasterio wheel. The PostgreSQL/PostGIS installer sets one on
Windows.

`hypersat` detects this before doing any CRS work: it probes EPSG:4326 and, if the probe
fails, falls back to the database bundled with rasterio. The fallback is not silent — it
logs a warning naming the conflicting path, and `hypersat validate` reports it as a
`proj_database` warning. Pass `--no-proj-autofix` (or set `HYPERSAT_PROJ_AUTOFIX=0`) to
leave the environment untouched and let the failure surface instead.

To fix it properly, clear both variables in the shell that runs `hypersat`. Note that
*re-pointing* them at the bundled directory does not work, because PROJ resolves its search
path when the library loads; the in-process fallback above uses
`rasterio.env.set_proj_data_search_path()`, which updates that path afterwards.

### Optional: the `osgeo.gdal` bindings

PyPI publishes the `gdal` package as a **source distribution only**, so `pip install gdal`
needs a matching system libgdal and a C++ toolchain — it will not build on a bare Windows
machine. They are therefore an optional extra:

```bash
pip install -e ".[dev,gdal]"     # only in an environment that already provides libgdal
```

**No code uses them today.** They were kept as an escape hatch for metadata rasterio might
not expose, but rasterio 1.5 covers everything the inspection stage needs — including
listing metadata domains (`tag_namespaces()`) and reading the RPC sensor model
(`dataset.rpcs`). `hypersat inspect` reports whether the bindings are installed, and that
is all it does with them.

## Docker

```bash
make docker-build                # or: docker build -t hypersat-processing-pipeline:dev .
docker run --rm hypersat-processing-pipeline:dev version

docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/outputs:/app/outputs" \
  hypersat-processing-pipeline:dev inspect --input data/raw/<product>
```

The image is based on `python:3.12-slim` plus rasterio wheels, so the container and a
local `pip install` behave identically.

## CLI examples

```bash
# Version and runtime information
hypersat version
hypersat version --json
hypersat version --verbose          # also rasterio / GDAL / PROJ versions

# Inspect a product directory or a raster: dimensions, bands, dtype, NoData, CRS,
# transform, bounds, resolution, driver, file size, metadata domains, RPC availability,
# per-band descriptions and wavelengths. Metadata only - no pixels are read.
hypersat inspect --input data/raw/enmap_l1b_product
hypersat inspect --input data/samples/scene.tif --json > inspection.json
hypersat inspect --input data/samples/scene.tif --bands 1,42,120 --checksum
hypersat inspect --input data/samples/scene.tif --band-limit 0   # print every band

# Pre-flight validation. Every check runs, so all problems appear in one pass; the
# process then exits with the code of the first blocking finding.
hypersat validate --input data/raw/enmap_l1b_product \
                  --require-rpc \
                  --require-wavelengths \
                  --dem data/dem/dem_cop30.tif \
                  --output-dir outputs/enmap_demo

hypersat validate --input data/samples/scene.tif --json
hypersat validate --input data/samples/scene.tif --strict   # warnings become failures

# Cosmetic PNG preview. Never writes a GeoTIFF and never changes scientific values.
# true-color / false-color bands are chosen by wavelength; pass --bands to override.
hypersat preview --input data/samples/scene.tif --output-dir outputs/preview
hypersat preview --input data/samples/scene.tif --output-dir outputs/preview \
                 --composite false-color
hypersat preview --input data/samples/scene.tif --output-dir outputs/preview \
                 --composite band --band 42 --max-dimension 1024
hypersat preview --input data/samples/scene.tif --output-dir outputs/preview \
                 --bands 40,30,20 --blur-kernel 3 --overwrite
```

`--json` writes to stdout while logs and diagnostics go to stderr, so a `--json` run can be
piped straight into another tool.

Global options: `--log-level` (`DEBUG`..`CRITICAL`) and `--log-format` (`text`/`json`),
also settable via `HYPERSAT_LOG_LEVEL` / `HYPERSAT_LOG_FORMAT`.

### Exit codes

The CLI is meant to be scripted, so exit codes are a tested contract:

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Unexpected internal error |
| 2 | CLI usage error |
| 3 | Configuration error |
| 4 | Product / input validation error |
| 5 | Raster I/O error |
| 6 | Processing error |
| 7 | Pipeline orchestration error |
| 8 | Missing optional dependency |
| 9 | Requested functionality not implemented yet |

### Library API

Reading, band selection and writing are available as a library before they get their own
commands (the preview and index commands arrive in milestones 4 and 5):

```python
from pathlib import Path

from hypersat.analytics.bands import true_colour_bands
from hypersat.io.inspect import inspect_raster
from hypersat.io.reader import ReadOptions, read_chunk
from hypersat.io.writer import write_chunk
from hypersat.models.raster import ReadWindow

scene = Path("data/samples/scene.tif")
info = inspect_raster(scene)

# Bands are chosen by wavelength, not by an index tied to one sensor's layout.
red, green, blue = true_colour_bands([band.wavelength_nm for band in info.bands])

chunk = read_chunk(
    scene,
    window=ReadWindow(col_off=0, row_off=0, width=512, height=512),
    bands=[red.index, green.index, blue.index],
    options=ReadOptions(as_float32=True),
)

# CRS, transform, NoData, descriptions and wavelengths travel with the pixels. The RPC
# sensor model does not, because this chunk is a subset of the source grid.
write_chunk(Path("outputs/subset.tif"), chunk)
```

Omitting `window` reads the whole raster, which on a real hyperspectral product raises
`MemoryBudgetExceededError` rather than allocating gigabytes; `iter_chunks` walks the
file's own block grid instead.

## Data acquisition

No data is committed, and the pipeline never downloads anything by itself.

* **Imagery:** an EnMAP **L1B** product from DLR's EOWEB GeoPortal
  (<https://eoweb.dlr.de>, registration required). L1B is the interesting input because it
  is in *sensor geometry* with RPC coefficients — an L1C is already orthorectified. Put it
  in `data/raw/`.
* **DEM:** Copernicus DEM GLO-30 (via <https://opentopography.org> or the AWS Open Data
  registry) covering the scene with a margin. Put it in `data/dem/`.

Full instructions, alternatives and the vertical-datum caveat:
[docs/data-sources.md](docs/data-sources.md).

## Example outputs

Nothing here yet. Example imagery, previews and a real `qc_report.json` will be added in
milestone 10, generated from a real EnMAP scene and labelled with the exact processing that
produced them. This section stays empty rather than showing synthetic data that could be
mistaken for a real satellite-processing result.

## Georeferencing vs reprojection vs orthorectification

These are three different operations, and conflating them is how geometrically wrong
products get published:

* **Georeferencing** associates imagery with geographic coordinates — either an affine
  transform plus CRS, or a *sensor model* (RPC coefficients) for imagery that is not yet on
  a map grid. An EnMAP L1B product is the second kind.
* **Reprojection** re-expresses an image that is *already* on a map grid in a different
  CRS. It resamples pixels but uses no terrain data and no sensor model, so it cannot
  remove relief displacement.
* **Orthorectification** corrects geometric displacement caused by viewing geometry and
  terrain, using a **sensor model** *and* a **DEM**. For every output map pixel it looks up
  the ground height, evaluates the RPC polynomials to find the detector sample that
  observed that location, and resamples there.

Terrain relief displaces a feature by roughly `h * tan(theta)` — about 88 m for a 500 m
hill viewed 10° off nadir, roughly three EnMAP pixels. That error is why the DEM is
mandatory.

**The pipeline never silently substitutes reprojection for orthorectification.** Missing
RPC metadata raises `MissingRPCMetadataError`; a missing or unreadable DEM raises
`MissingDEMError` / `UnreadableDEMError`. Full explanation, including the RPC model itself:
[docs/orthorectification.md](docs/orthorectification.md).

## Quality-mask classes

Single-band `uint8`, one class per pixel, built in sensor geometry
([docs/quality-masks.md](docs/quality-masks.md), implementation in milestone 6):

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | `NO_DATA` | Outside the acquisition or equal to the source NoData value |
| 1 | `VALID` | Nominal observation |
| 2 | `SATURATED` | Detector saturation in the evaluated bands |
| 3 | `LOW_SIGNAL` | Deep shadow, sensor floor, very low radiance |
| 4 | `INVALID_NUMERIC` | NaN or infinite sample |
| 5 | `SPECTRAL_ANOMALY` | Optional plausibility check, disabled by default |
| 255 | `UNCLASSIFIED` | Pixel was not evaluated |

Precedence, most severe first:
`NO_DATA > INVALID_NUMERIC > SATURATED > LOW_SIGNAL > SPECTRAL_ANOMALY > VALID`.
Class `0` is also the mask's NoData value, so nearest-neighbour warping fills the area
outside the swath with a code that already means "no data".

There is deliberately **no cloud mask**: honest cloud screening needs reflectance-space
physics or a validated classifier, not ad-hoc radiance thresholds.

## Configuration

`configs/pipeline.example.yaml` is the annotated reference for the schema the pipeline is
growing into. Today's commands are driven by CLI options rather than by that file: YAML
loading and the `hypersat process` runner arrive in milestone 9, and only the parts of the
schema a working stage actually reads exist as models so far.

What is already true is the modelling style. Every configuration model is a frozen Pydantic
model that **rejects unknown keys**, so a mistyped option is an error rather than a
silently ignored setting, and validation failures surface as `ConfigurationError` with exit
code 3.

## Testing

```bash
make test               # unit + integration; `external` tests excluded by default
make test-integration   # integration tests only
make test-external      # opt-in: needs a real product/DEM that is not in the repo
make lint               # ruff check + ruff format --check
make type-check         # mypy --strict
make check              # all of the above, in CI order
```

Without make: `pytest`, `ruff check src tests`, `ruff format --check src tests`, `mypy`.

Test fixtures are **tiny GeoTIFFs generated in-process** with rasterio. They are test
inputs only and are never presented as processing results. A green suite proves the code
behaves as specified; it does not prove scientific correctness of a product derived from
real mission data.

## Scientific limitations

Summarised — the full list is in [docs/limitations.md](docs/limitations.md):

* No radiometric calibration, no spectral smile/keystone correction, no bad-pixel
  interpolation.
* No atmospheric correction. Output is at-sensor radiance, never surface reflectance,
  so NDVI/NDWI here are arithmetic on radiance and are not comparable across scenes.
* Geometric accuracy is bounded by the delivered RPC model: no GCPs, no bundle adjustment,
  no co-registration, no vertical datum conversion, and no accuracy figure is reported
  because that requires independent reference data.
* No cloud, shadow, snow or water detection.
* Output is never an official mission product level.

## Software engineering decisions

* **Layered architecture with one geospatial boundary.** Only `hypersat.io` (and the
  geometric processing modules) touch rasterio/GDAL; analytics and visualisation work on
  NumPy arrays plus explicit metadata, so pixel maths is unit-testable without sample
  products.
* **Pure functions for raster mathematics**, classes only where state genuinely exists.
* **Errors are a typed hierarchy with exit codes**, each carrying an actionable `hint` and
  structured `context` that goes straight into the logs — and into the QC report once that
  exists.
* **Validation aggregates instead of aborting.** `hypersat validate` runs every check and
  reports all findings in one pass, then exits with the code of the most specific blocking
  one. An operator preparing a long orthorectification run should not discover the
  unwritable output directory only after fixing the missing DEM.
* **Configuration as explicit frozen Pydantic models** that reject unknown keys, not a dict
  of magic keys.
* **Structured logging** with `--log-format json` for container and CI log collectors, and
  machine-readable results on stdout with diagnostics on stderr.
* **`pathlib` throughout**, enforced by Ruff's `PTH` rules; no hardcoded local paths and
  no hardcoded EPSG codes outside examples and tests.
* **Wavelengths are read from metadata, with the source recorded per band**, so band
  selection is driven by wavelength rather than by an index tied to one sensor's layout.
  `hypersat.analytics.bands` resolves a wavelength to the nearest band within a tolerance,
  breaks ties towards the lower index, and reports the distance so a poor-but-legal match
  can be surfaced instead of hidden.
* **Atomic writes** for every raster output — write to a temporary sibling, move into
  place with `Path.replace` only after the dataset closes cleanly — so a crash cannot
  leave a plausible-looking partial product.
* **Reads are budgeted.** A full hyperspectral cube does not fit in a comfortable amount
  of memory, so the reader states the cost of a read before allocating and refuses one
  that exceeds the budget, pointing at windowed or band-selective reading instead.
* **Strict quality gates:** `ruff` (including pydocstyle and flake8-annotations), `mypy
  --strict` over `src` *and* `tests`, `pytest` on Python 3.11-3.13, and a Docker build, all
  in CI.
* **No hidden network access** inside processing functions; data acquisition is an explicit
  documented step.
* **Deliberately absent:** Django, FastAPI, Celery, Airflow, Kubernetes, a frontend and
  PyTorch. None of them would make the geospatial problem better solved.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the milestone plan and the open decisions
(notably the warp backend choice for milestone 8).

## License

MIT — see [LICENSE](LICENSE).
