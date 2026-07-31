"""Domain models: Pydantic configuration schemas and processing-report structures.

This layer is pure data. It must not import rasterio, OpenCV or the GDAL bindings, so
that configuration files can be validated without a geospatial runtime present.

Modules:

* ``base`` - ``StrictModel``, the shared Pydantic configuration (implemented).
* ``config`` - per-command option models (implemented, grows per milestone).
* ``product`` - ``RasterInfo``, ``BandInfo``, ``RPCInfo`` inspection results (implemented).
* ``environment`` - runtime and PROJ-database description (implemented).
* ``validation`` - validation checks and reports (implemented).
* ``raster`` - ``ReadWindow``, ``RasterChunk``, ``RasterMetadata`` (implemented). These are
  frozen dataclasses rather than Pydantic models because they carry NumPy arrays; see the
  module docstring.
* ``report`` - ``QualityControlReport``, ``StageReport``, ``BandStatistics`` (milestone 10).
"""
