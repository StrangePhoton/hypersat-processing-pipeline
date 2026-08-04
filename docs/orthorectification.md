# Georeferencing, reprojection and orthorectification

These three terms are routinely mixed up. This project keeps them separate in code,
naming and documentation, because conflating them is exactly how geometrically wrong
products get published.

## 1. The three operations

### Georeferencing — "where is this image on Earth?"

Georeferencing attaches a geographic meaning to image coordinates. It can take two very
different forms:

* **Affine georeferencing.** A CRS plus a 6-parameter affine transform mapping
  `(column, row)` to `(x, y)`. Cheap, exact for an already-map-projected grid, and what
  `rasterio`'s `dataset.crs` and `dataset.transform` express.
* **A sensor model.** For raw or geometrically uncorrected imagery there is no single
  affine transform, because the mapping from pixel to ground depends on satellite
  position, attitude, sensor viewing geometry and terrain height. Instead the product
  carries a model: RPC coefficients, or a rigorous physical model with orbit/attitude
  data.

An EnMAP L1B product is in the second category. It is *georeferenced* in the sense that
it carries a sensor model and geographic corner coordinates, but its pixels are **not**
on a map grid.

### Reprojection — "express the same grid in another CRS"

Reprojection takes an image that is already on a map grid and re-expresses it on a grid
in a different CRS, for example EPSG:4326 to EPSG:32633. It resamples pixels, so it
changes pixel values, but it applies **no terrain knowledge and no sensor model**. It
cannot remove relief displacement; if the input was geometrically distorted, the output
is distorted the same way in new coordinates.

### Orthorectification — "remove geometric displacement caused by viewing geometry and terrain"

Orthorectification computes, for every output map pixel, which detector sample actually
observed that ground location, using:

1. a **sensor model** (here: RPC coefficients), and
2. a **digital elevation model** to know the ground height at that location.

Both inputs are mandatory. Terrain relief displaces a feature in the image by roughly
`h * tan(theta)`, where `h` is height above the reference surface and `theta` is the
off-nadir viewing angle. For a 500 m hill viewed 10 degrees off nadir, that is about
88 m of displacement — roughly three EnMAP pixels. Without a DEM, that error remains.

| Operation | Needs sensor model | Needs DEM | Corrects relief displacement |
| --- | --- | --- | --- |
| Georeferencing (affine) | No | No | No |
| Reprojection | No | No | No |
| RPC warp without DEM | Yes | No | No (only a constant-height approximation) |
| Orthorectification | Yes | Yes | Yes |

## 2. What the RPC model is

The Rational Polynomial Coefficients model is a generic, sensor-agnostic replacement for
a physical camera model. It expresses normalised image coordinates as ratios of cubic
polynomials in normalised longitude, latitude and height:

```
row_n = P1(lon_n, lat_n, h_n) / P2(lon_n, lat_n, h_n)
col_n = P3(lon_n, lat_n, h_n) / P4(lon_n, lat_n, h_n)
```

Each `Pi` is a 20-term cubic polynomial, so a full RPC set is 80 coefficients plus 10
offset/scale normalisation values (`LINE_OFF`, `SAMP_OFF`, `LAT_OFF`, `LONG_OFF`,
`HEIGHT_OFF`, and the matching `_SCALE` terms). GDAL exposes them in the `RPC` metadata
domain; rasterio exposes them as `dataset.rpcs`, an `rasterio.rpc.RPC` object with
`to_dict()`/`to_gdal()`.

### How GDAL normalises the RPC domain

Verified against GDAL 3.12.1 / rasterio 1.5.0 while implementing inspection, because it
determines what "incomplete RPC" can even mean:

* An RPC set that is **missing required keys is discarded entirely** — `dataset.rpcs`
  returns `None`. A partially written model therefore appears as "no sensor model at all",
  not as a broken one.
* A coefficient list **shorter than 20 terms is silently zero-padded** to 20. Counting
  terms can therefore never detect a truncated product on its own.
* Absent optional scalars are filled with defaults (`ERR_BIAS`/`ERR_RAND` become `-1`, a
  missing `HEIGHT_OFF` becomes `0`).

Consequently `hypersat inspect` does not claim to verify RPC "completeness" by counting.
It reports `available`, and an `is_usable` flag that looks for values which would make the
transformation undefined: an all-zero polynomial (what a truncated set degrades into) or a
zero normalisation scale (which would divide by zero during normalisation). Anything
beyond that requires evaluating the model against reference data, which this project does
not do.

Note the direction: the model maps **ground to image**. Warping therefore works by
inverse mapping — for each output map pixel, convert to lon/lat, look up the height in
the DEM, evaluate the RPC to obtain a fractional image coordinate, and resample the
source there. This is why a DEM is not an optional refinement: the height is an *input*
to the coordinate transformation, not a post-hoc correction.

RPC accuracy is limited by the coefficients themselves. Without ground control points or
bundle adjustment, the absolute geolocation accuracy of the output is bounded by the
accuracy of the delivered RPCs (metres to tens of metres, mission-dependent). This
project does **not** implement GCP-based refinement, so it cannot improve on the
delivered model.

## 3. How this project implements it

The orthorectification stage is implemented (`hypersat orthorectify` /
`hypersat.processing.orthorectification`) and does the following:

1. **Verify the sensor model exists.** If the raster has no RPC metadata, raise
   `MissingRPCMetadataError`. It never falls back to plain reprojection.
2. **Verify the DEM.** Missing path raises `MissingDEMError`; an unreadable DEM or one
   that does not cover the scene raises `UnreadableDEMError`. Running "orthorectification"
   with a DEM that does not overlap the scene silently degrades to a constant-height
   warp, so the overlap check is mandatory, not an optional nicety.
3. **Choose the target CRS.** Either an explicit authority code, or `auto`, which picks
   the UTM zone containing the scene centre (and refuses when the scene straddles zones
   too widely or lies beyond the UTM latitude limits, where a polar stereographic CRS
   would be appropriate instead).
4. **Warp with GDAL via rasterio** using the RPC transformer with the DEM supplied via
   the `RPC_DEM` transformer option, the configured output resolution and resampling.
5. **Resample by data semantics.** Continuous imagery uses `bilinear` or `cubic`;
   categorical rasters (the quality mask) always use `nearest`, because averaging class
   codes produces meaningless intermediate values.
6. **Preserve NoData** explicitly on both the source and destination side, so that areas
   outside the swath are NoData rather than zero-valued "real" data.
7. **Write tiled, compressed GeoTIFF** output, written atomically.
8. **Record the exact configuration** — transformer options, resampling kernel, DEM path,
   target CRS, output grid — in the output GeoTIFF tags (and later the QC report), so a
   reviewer can tell what was actually done.

### Backend: rasterio over `osgeo.gdal.Warp`

Both APIs call the same GDAL C++ code (`GDALCreateRPCTransformer` plus the warping
kernel). This project uses **`rasterio.warp.reproject(..., rpcs=..., RPC_DEM=...)`**
because rasterio wheels bundle libgdal and install with `pip` on Linux, macOS and
Windows. `osgeo.gdal` remains an optional extra (`pip install -e ".[gdal]"`) for
environments that already ship a matching system libgdal. Transformer options passed to
GDAL are echoed into the output tags.

## 4. What this stage does not do

* No ground control points, tie points or bundle adjustment.
* No sub-pixel co-registration against a reference image.
* No DEM void filling, vertical datum conversion (geoid to ellipsoid) or DEM resampling
  quality analysis. If a DEM's vertical reference differs from the one the RPCs assume,
  a systematic horizontal error remains, and the report states the DEM used rather than
  claiming the correction was validated.
* No orthorectification of areas the DEM does not cover.
* No claim of a specific geolocation accuracy. The project reports what it did; it does
  not certify how good the result is, because that requires independent reference data.
