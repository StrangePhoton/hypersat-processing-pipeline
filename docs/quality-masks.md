# Quality mask specification

The quality mask is a single-band `uint8` GeoTIFF built in **sensor geometry**, before
orthorectification (rationale in `docs/architecture.md`). Implemented in milestone 6 via
`hypersat quality-mask` / `hypersat.processing.quality_mask`. This document is the class
contract that reports, tests and downstream code rely on.

## 1. Class codes

Each pixel receives exactly one class code.

| Code | Name | Meaning | How it is decided |
| --- | --- | --- | --- |
| 0 | `NO_DATA` | Outside the acquisition, fill value, or masked by the source NoData | Sample equals the raster NoData value in the evaluated bands |
| 1 | `VALID` | Nominal observation, no problem detected | No other class applies |
| 2 | `SATURATED` | Detector saturation; radiometry unusable | DN >= `saturation_dn` in at least `saturation_band_fraction` of the evaluated bands |
| 3 | `LOW_SIGNAL` | Very low signal: deep shadow, water at long wavelengths, sensor floor | DN <= `low_signal_dn` in at least `saturation_band_fraction` of the evaluated bands |
| 4 | `INVALID_NUMERIC` | NaN or infinite value in a floating-point input | `~numpy.isfinite(sample)` |
| 5 | `SPECTRAL_ANOMALY` | Optional: spectrum fails a configured plausibility check | Only when explicitly enabled; disabled by default |
| 255 | `UNCLASSIFIED` | Reserved: the mask exists but this pixel was not evaluated | e.g. band subset excluded the evaluation bands |

`0` is deliberately both `NO_DATA` and the raster's NoData value. Nearest-neighbour
warping then fills the area outside the swath with `0`, which already means "no data" —
so the orthorectified mask needs no special post-processing to stay consistent.

## 2. Class precedence

A pixel can satisfy several conditions at once (a NaN sample in a saturated band, for
example). Classes are therefore assigned in a fixed precedence order, most severe first:

```
NO_DATA > INVALID_NUMERIC > SATURATED > LOW_SIGNAL > SPECTRAL_ANOMALY > VALID
```

Rationale: a pixel that carries no data cannot be saturated, and a numerically invalid
sample must never be reported as merely "low signal", because a downstream consumer
filtering on `VALID` would otherwise ingest a NaN.

## 3. Thresholds are configuration, not constants

`saturation_dn`, `low_signal_dn`, the evaluated wavelengths and the band fraction are all
configuration values (see `configs/pipeline.example.yaml`), because the correct saturation
level is a property of the instrument and product encoding, not of this software. The
defaults in the example configuration are illustrative values for a 16-bit DN product and
are **not** a mission-validated saturation specification.

Bands to evaluate are selected by **wavelength**, and the actually selected band indices
and their true centre wavelengths are recorded in the QC report.

## 4. Morphological post-processing

OpenCV morphology (`open`, `close`, `dilate`, `erode`) is available but **disabled by
default**. When enabled, the operation, kernel shape, kernel size and iteration count are
all explicit configuration values, and all of them are echoed into the QC report.

Morphology is applied only to *defect* classes, never to reclassify a pixel as `VALID`:
growing a saturation region is a conservative choice, whereas shrinking one would hide
bad data.

## 5. Reported statistics

The QC report contains, for the mask in both geometries:

* the pixel count and percentage per class,
* `valid_percentage` (class 1),
* `nodata_percentage` (class 0),
* `saturated_percentage` (class 2),
* the mask NoData value and the resampling used when warping it.

## 6. Deliberate limitations

* **No cloud, cirrus or shadow detection.** Reliable cloud screening needs either
  reflectance-space thresholds with atmospheric parameters or a trained classifier, plus
  validation data. Claiming a cloud mask from L1B radiance with ad-hoc thresholds would
  be exactly the kind of misleading shortcut this project avoids.
* **No snow/ice, water or land-cover classification.**
* **No per-band bad-pixel or dead-column map**, because that requires the instrument's
  bad-pixel table.
* **One class per pixel, not a bit field.** A bit-flag layer (allowing multiple
  simultaneous flags) is a roadmap item; the single-class scheme keeps the MVP's
  precedence rules explicit and testable.
