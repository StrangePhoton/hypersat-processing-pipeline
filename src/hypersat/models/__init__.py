"""Domain models: Pydantic configuration schemas and processing-report structures.

This layer is pure data. It must not import rasterio, OpenCV or the GDAL bindings, so
that configuration files can be validated without a geospatial runtime present.

Planned modules (milestone 2 onwards):

* ``config`` - ``PipelineConfig`` and per-stage option models.
* ``product`` - ``RasterInfo``, ``BandInfo``, ``RPCInfo`` inspection results.
* ``report`` - ``QualityControlReport``, ``StageReport``, ``BandStatistics``.
"""
