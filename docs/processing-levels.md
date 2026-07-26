# Processing levels and honest product labelling

## 1. Standard Earth-observation processing levels

The levels below follow common EO practice (and EnMAP's own product ladder). They are
summarised here so it is clear which parts of the chain this project touches.

| Level | Name | What it contains | Who produces it |
| --- | --- | --- | --- |
| L0 | Raw | Decoded telemetry, instrument packets, no radiometric meaning | Mission ground segment |
| L1A | Uncalibrated, unpacked | Detector digital numbers, framed and annotated | Mission ground segment |
| L1B | Radiometrically calibrated, sensor geometry | At-sensor radiance (or DN plus calibration), **not** on a map grid; carries a sensor model (RPCs) and per-band wavelengths | Mission ground segment |
| L1C | Orthorectified / map projected | Radiance resampled onto a map grid using the sensor model, a DEM and mission-internal geometric calibration | Mission ground segment |
| L2A | Surface reflectance | Atmospherically corrected bottom-of-atmosphere reflectance, plus cloud/quality masks | Mission ground segment |

Key point for hyperspectral data: L1B to L1C is a **geometric** step, L1C to L2A is a
**radiometric/atmospheric** step. They are independent, and doing one does not imply the
other.

## 2. Where this project operates

```
L0 ---- L1A ---- L1B ----------> [ THIS PROJECT ] ----> demonstration products
                  ^                                          |
                  |                                          +-- geometric processing:
            provided input                                    |   RPC + DEM orthorectification,
            (publicly downloadable)                           |   reprojection, grid alignment
                                                              +-- quality masking
                                                              +-- spectral indices
                                                              +-- QC reporting
```

The project starts from a **published L1B product** and performs the geometric and
quality-related parts of the downstream chain. It does not perform:

* raw telemetry decoding or detector framing (L0/L1A),
* radiometric calibration — dark current, gain/offset, stray light, spectral smile,
  keystone or bad-pixel interpolation,
* atmospheric correction (aerosol retrieval, water-vapour retrieval, adjacency effects,
  BRDF or topographic illumination correction),
* mission-internal geometric calibration or GCP-based bundle adjustment.

Those steps require mission-specific calibration data, auxiliary atmospheric products and
validated algorithms that are not part of a public product download.

## 3. Product labelling rules

Because the project performs *only* the geometric part, calling its output "L1C" would be
false: an official L1C also embodies mission geometric calibration and an operationally
validated processing chain. The following naming rules are therefore enforced.

**Never used:**

* `L1C`, `L2A`, `L2B`, or any other mission product-level token in filenames, logs,
  reports or documentation, for output this project generates.
* "calibrated", "atmospherically corrected", "surface reflectance", "analysis-ready
  data (ARD)" as unqualified claims.

**Used instead:**

| Output | Correct label |
| --- | --- |
| Orthorectified radiance cube | "orthorectified demonstration product" |
| Reprojected / grid-aligned raster | "reprojected demonstration product" |
| NDVI / NDWI raster | "spectral index computed from L1B radiance, not from surface reflectance" |
| The overall result set | "analysis-ready demonstration output" |

Filenames encode only the operation performed and the output grid, e.g.
`<id>_ortho_epsg32633_30m.tif` (see the naming convention in `docs/architecture.md`).

The QC report records `processing_performed` as an explicit list of the operations
actually executed, plus `not_performed`, listing radiometric calibration and atmospheric
correction. Any downstream consumer reading the report can therefore see the boundary
without reading the documentation.

## 4. Consequence for spectral indices

NDVI and NDWI are conventionally computed from surface reflectance. Computed from L1B
at-sensor radiance they are still well-defined arithmetic, but they are **not** comparable
to reflectance-based NDVI from another scene, date or sensor, because atmospheric path
radiance, illumination geometry and per-band solar irradiance all remain in the signal.

The pipeline therefore:

* records the source of the bands (radiance, not reflectance) in the QC report,
* records the actual centre wavelength of each selected band, not just the nominal one,
* labels index outputs as demonstration products,
* does not resolve the values as vegetation-health measurements anywhere in the docs.

NDVI and NDWI are also not hyperspectral algorithms. They are two-band normalised
differences that happen to be computed here from bands selected out of a hyperspectral
cube by wavelength. Genuinely hyperspectral analytics (continuum removal, spectral
unmixing, absorption-feature fitting, matched filtering) are listed in the roadmap and
are honestly absent from the MVP.
