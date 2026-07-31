"""Raster and product I/O: the only layer allowed to touch rasterio or GDAL directly.

Keeping every dataset open/read/write behind this boundary means the processing and
analytics layers operate on NumPy arrays plus explicit geospatial metadata, which makes
them unit-testable without sample products on disk.

Modules:

* ``inspect`` - read-only product/raster introspection (implemented).
* ``files`` - sizes, timestamps and checksums (implemented).
* ``environment`` - geospatial runtime diagnostics and PROJ fallback (implemented).
* ``reader`` - windowed, band-selective, memory-budgeted reading with masked arrays
  (implemented).
* ``writer`` - atomic, tiled, compressed GeoTIFF writing (implemented).
"""
