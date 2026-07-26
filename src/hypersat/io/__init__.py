"""Raster and product I/O: the only layer allowed to touch rasterio or GDAL directly.

Keeping every dataset open/read/write behind this boundary means the processing and
analytics layers operate on NumPy arrays plus explicit geospatial metadata, which makes
them unit-testable without sample products on disk.

Planned modules (milestone 2 onwards):

* ``inspect`` - read-only product/raster introspection.
* ``reader`` - windowed, band-selective, memory-aware reading with masked arrays.
* ``writer`` - atomic, tiled, compressed GeoTIFF writing.
* ``product`` - EnMAP product-directory discovery (metadata XML plus raster locations).
"""
