"""Spectral analytics implemented as pure functions over NumPy arrays.

Nothing in this layer opens files; callers pass arrays, wavelengths and NoData values.

Planned modules (milestone 6 onwards):

* ``bands`` - wavelength-to-band-index selection.
* ``indices`` - normalised-difference indices (NDVI, NDWI) with safe division.
* ``profiles`` - per-pixel spectral profile extraction.
* ``statistics`` - per-band descriptive statistics honouring NoData and non-finite values.
"""
