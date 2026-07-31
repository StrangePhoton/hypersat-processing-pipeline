"""Spectral analytics implemented as pure functions over NumPy arrays.

Nothing in this layer opens files; callers pass arrays, wavelengths and NoData values.

Modules:

* ``bands`` - wavelength-to-band-index selection (implemented).
* ``indices`` - normalised-difference indices (NDVI, NDWI) with safe division (implemented).
* ``profiles`` - per-pixel spectral profile extraction (implemented).
* ``statistics`` - per-band descriptive statistics honouring NoData and non-finite values
  (implemented).
"""
