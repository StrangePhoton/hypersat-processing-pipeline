# Architecture

## 1. Design goals

1. **Scientific honesty over feature count.** A stage either does the real thing or
   refuses to run. Nothing is approximated silently.
2. **Testable pixel maths.** All array mathematics lives in pure functions that take
   NumPy arrays and explicit metadata, so tests need no sample products.
3. **One boundary for geospatial libraries.** Only `hypersat.io` and
   `hypersat.processing` may import rasterio/GDAL. Everything else works on arrays,
   Pydantic models and `pathlib.Path`.
4. **Bounded memory.** A full EnMAP cube (about 1000 x 1000 x 224 samples) is never
   loaded implicitly. Readers are windowed and band-selective.
5. **Reproducible runs.** Every run writes a JSON quality-control report containing the
   effective configuration, library versions, parameters and output checksums.

## 2. Layers

```
                        +-------------------------------+
                        |          hypersat.cli         |  Typer commands, thin
                        |  parse args -> call service   |  handlers, exit codes
                        +---------------+---------------+
                                        |
                        +---------------v---------------+
                        |       hypersat.pipeline       |  stage protocol, runner,
                        |  order, timing, QC report     |  registry, artifacts
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
          |  inspect, files, environment, reader, writer           |  that opens files
          +----------+--------------------------------------------+
                     |
          +----------v--------------------------------------------+
          |        rasterio / GDAL / optional osgeo.gdal           |
          +-------------------------------------------------------+

          hypersat.models        pure Pydantic data, no I/O, no geospatial imports
          hypersat.exceptions    error hierarchy + exit-code contract
          hypersat.logging_config structured text/JSON logging
```

Dependency rule: arrows point downwards only. `models` and `exceptions` may be imported
by anything; they import nothing from the package.

## 3. Module map

| Module | Responsibility | Milestone |
| --- | --- | --- |
| `hypersat.exceptions` | Error hierarchy, exit codes | 1 (done) |
| `hypersat.logging_config` | Text/JSON structured logging | 1 (done) |
| `hypersat.cli` | Typer commands, error-to-exit-code mapping | 1 skeleton, grows per milestone |
| `hypersat.formatting` | Pure model-to-text renderers for console output | 2 (done) |
| `hypersat.models.base` | `StrictModel`: forbid extras, frozen, JSON-safe NaN | 2 (done) |
| `hypersat.models.config` | Configuration models for implemented commands | 2 (done), grows per milestone |
| `hypersat.models.product` | `RasterInfo`, `BandInfo`, `RPCInfo`, `CRSInfo`, `FileInfo`, `ProductLayout`, `InspectionResult` | 2 (done) |
| `hypersat.models.environment` | `EnvironmentInfo`, `ProjDataStatus` | 2 (done) |
| `hypersat.models.validation` | `ValidationCheck`, `ValidationReport` | 2 (done) |
| `hypersat.models.report` | `QualityControlReport`, `StageReport`, `BandStatistics` | 10 |
| `hypersat.io.inspect` | Read-only raster/product introspection, product-directory scan | 2 (done) |
| `hypersat.io.files` | Sizes, timestamps, SHA-256 checksums | 2 (done) |
| `hypersat.io.environment` | Version reporting, PROJ database probe and fallback | 2 (done) |
| `hypersat.io.reader` | Windowed, band-selective, masked reading | 3 |
| `hypersat.io.writer` | Atomic tiled/compressed GeoTIFF writing | 3 |
| `hypersat.processing.validation` | Pre-flight product/DEM/output checks | 2 (done) |
| `hypersat.processing.quality_mask` | Quality-class raster, OpenCV morphology | 6 |
| `hypersat.processing.reprojection` | CRS transform, alignment, resampling choice | 7 |
| `hypersat.processing.orthorectification` | RPC + DEM warp | 8 |
| `hypersat.analytics.bands` | Wavelength-to-index selection | 3 |
| `hypersat.analytics.indices` | NDVI, NDWI with safe division | 5 |
| `hypersat.analytics.profiles` | Pixel spectral profiles | 5 |
| `hypersat.analytics.statistics` | Per-band descriptive statistics | 5 |
| `hypersat.visualization.stretch` | Percentile stretching | 4 |
| `hypersat.visualization.preview` | RGB / false-colour / single-band PNG | 4 |
| `hypersat.pipeline.stage` | `Stage` protocol, `StageContext`, `StageResult` | 9 |
| `hypersat.pipeline.runner` | Sequential executor, timing, failure isolation | 9 |
| `hypersat.pipeline.registry` | Config key to stage mapping | 9 |

A planned `hypersat.io.product` module was dropped in milestone 2: locating the imagery
inside a product directory turned out to be ~40 lines of path scanning that belongs next to
the inspection code it serves, and a separate module would have been an empty layer. If
mission-specific metadata parsing (EnMAP's `*-METADATA.XML`) is added later, it gets its
own module then - when there is something to put in it.

## 4. Processing flow

```
  EnMAP L1B product (or GeoTIFF)
            |
   [1] validate         metadata only, no pixel reads
            |           checks: files, readability, size, bands, NoData,
            |           CRS or RPC, wavelengths, DEM, output path
            v
   [2] quality_mask     built in SENSOR geometry, on original detector samples
            |           -> <id>_qmask.tif (uint8, NoData = 0)
            v
   [3] orthorectify     RPC sensor model + DEM, GDAL warp
            |           imagery: nearest | bilinear | cubic
            |           mask: nearest ONLY
            |           -> <id>_ortho_epsg<code>_<res>m.tif
            |           -> <id>_qmask_ortho_epsg<code>_<res>m.tif
            v
   [4] align            optional: reproject/snap onto a reference grid
            |           -> <id>_aligned_epsg<code>_<res>m.tif
            v
   [5] spectral_indices bands chosen by wavelength, not by index number
            |           -> <id>_ndvi.tif, <id>_ndwi.tif
            v
   [6] preview          percentile-stretched 8-bit PNG, cosmetic only
            |           -> <id>_preview_<composite>.png
            v
   [7] report           always last, also written after a failure
                        -> qc_report.json
```

Why the mask comes before orthorectification: saturation and fill values are properties
of individual detector samples. After warping, an output pixel is a weighted mixture of
several input samples, so "was this sample saturated?" is no longer answerable. The mask
is therefore computed first and then warped with nearest-neighbour, which copies class
codes without inventing intermediate values.

## 5. Stage contract

Every stage satisfies one protocol (`hypersat.pipeline.stage`):

```python
class Stage(Protocol):
    name: str                       # stable identifier, matches the config key
    requires: tuple[str, ...]       # artifact keys that must already exist
    produces: tuple[str, ...]       # artifact keys this stage registers

    def run(self, context: StageContext) -> StageResult: ...
```

`StageContext` (read-only for the stage, except the artifact registry):

| Field | Meaning |
| --- | --- |
| `config` | The validated `PipelineConfig` |
| `run_id` | UTC timestamp + short random suffix, used in logs and the report |
| `output_dir` | Already-created, verified-writable output directory |
| `artifacts` | Mapping of artifact key to `Path`, populated by earlier stages |
| `logger` | Logger bound to the stage name |

`StageResult`:

| Field | Meaning |
| --- | --- |
| `name` | Stage name |
| `status` | `success` \| `skipped` \| `failed` |
| `duration_s` | Wall-clock duration, measured by the runner, not the stage |
| `outputs` | Artifact key to path mapping for files actually written |
| `metrics` | Numbers destined for the QC report (pixel percentages, offsets, ...) |
| `warnings` | Non-fatal issues, surfaced in logs and in the report |

Rules every stage must obey:

1. **Declared contract.** A stage reads only artifacts in `requires` and registers only
   artifacts in `produces`. The runner verifies both, so a misconfigured order fails
   before any pixels are touched.
2. **Log start and completion.** The runner emits `stage started` / `stage completed`
   with `extra={"stage": ..., "duration_s": ...}`; the stage logs its own parameters.
3. **Timing.** Measured by the runner with `time.perf_counter`, recorded per stage.
4. **Actionable failure.** Raise a `HyperSatError` subclass with a `hint`. The runner
   wraps it in `StageExecutionError`, keeping the original as `__cause__`.
5. **No misleading partial output.** Rasters are written to `<final>.tmp-<run_id>` in the
   destination directory and moved into place with `os.replace` only after the dataset is
   closed successfully. A crash therefore leaves no half-written GeoTIFF that a later run
   could mistake for a finished product.
6. **No hidden downloads.** Stages never fetch DEMs or products from the network. Data
   acquisition is a documented manual step (`docs/data-sources.md`).

## 6. Artifact keys

| Key | Produced by | Content |
| --- | --- | --- |
| `source_raster` | runner (from config) | Input raster, sensor geometry |
| `quality_mask` | `quality_mask` | uint8 class raster, sensor geometry |
| `ortho_raster` | `orthorectify` | Orthorectified cube, map geometry |
| `ortho_quality_mask` | `orthorectify` | Warped mask, nearest-neighbour |
| `aligned_raster` | `align` | Product snapped to a reference grid |
| `index:<name>` | `spectral_indices` | One float32 GeoTIFF per index |
| `preview:<name>` | `preview` | One PNG per composite |
| `band_statistics` | `report` | CSV of per-band statistics |
| `qc_report` | `report` | `qc_report.json` |

## 7. Output naming convention

```
<product_id>_<content>[_epsg<code>_<resolution>m].<ext>
```

* `product_id` - from `output.product_id`, or derived from the input name: lowercased,
  ASCII, non-alphanumeric runs collapsed to `_`.
* `content` - `ortho`, `qmask`, `qmask_ortho`, `aligned`, `ndvi`, `ndwi`,
  `preview_<composite>`, `band_statistics`, `spectral_profile`.
* The grid suffix is appended only to rasters in map geometry, so a filename always
  tells you the CRS and ground sample distance of the pixels inside it.
* Resolution uses `p` for the decimal separator: `0p5m`, `30m`.
* Extensions: `.tif` (GeoTIFF), `.png` (preview), `.csv` (tables), `.json` (report).
* `qc_report.json` keeps a fixed name so automation can find it without knowing the
  product id.

Example output directory:

```
outputs/enmap_demo/
├── enmap_l1b_20221103_qmask.tif
├── enmap_l1b_20221103_ortho_epsg32633_30m.tif
├── enmap_l1b_20221103_qmask_ortho_epsg32633_30m.tif
├── enmap_l1b_20221103_ndvi_epsg32633_30m.tif
├── enmap_l1b_20221103_preview_true_color.png
├── enmap_l1b_20221103_band_statistics.csv
└── qc_report.json
```

Nothing in a filename claims a mission processing level. There is no `_l1c` or `_l2a`
token, by design (see `docs/processing-levels.md`).

## 8. Cross-cutting decisions

* **Configuration is a model, not a dict.** `PipelineConfig` forbids extra keys, so a
  typo in a YAML key is an error rather than a silently ignored setting.
* **Exit codes are a contract.** Documented in `hypersat/exceptions.py` and covered by
  tests, because the CLI is meant to be scripted.
* **Structured logging.** `--log-format json` emits one JSON object per line with stage,
  duration and parameter fields as first-class keys.
* **No global mutable state.** Configuration is passed explicitly; logging setup is the
  only process-wide side effect and it is idempotent and confined to the `hypersat`
  logger namespace.
* **`pathlib` everywhere.** No string path concatenation, and Ruff's `PTH` rules enforce
  it.
* **Strict typing.** `mypy --strict` over `src` and `tests`; the package ships `py.typed`.
