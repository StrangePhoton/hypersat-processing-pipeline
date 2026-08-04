"""Geometric, quality and spectral processing stages.

Modules:

* ``validation`` - pre-flight product, DEM and output-path checks (implemented).
* ``spectral`` - index and profile orchestration over the analytics pure functions
  (implemented).
* ``quality_mask`` - quality-class raster and optional OpenCV morphology (implemented).
* ``reprojection`` - CRS transformation, grid alignment and resampling selection
  (milestone 7).
* ``orthorectification`` - RPC + DEM warping (milestone 8, see
  ``docs/orthorectification.md``).
"""
