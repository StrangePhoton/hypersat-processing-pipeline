"""Spectral analytics implemented as pure functions over NumPy arrays.

Nothing in this layer opens files; callers pass arrays, wavelengths and NoData values.

Planned modules:

* ``bands`` - wavelength-to-band-index selection (milestone 3, needed by the previews).
* ``indices`` - normalised-difference indices (NDVI, NDWI) with safe division (milestone 5).
* ``profiles`` - per-pixel spectral profile extraction (milestone 5).
* ``statistics`` - per-band descriptive statistics honouring NoData and non-finite values
  (milestone 5).
"""
