"""Preview rendering. Never writes scientific rasters, only 8-bit PNG previews.

Percentile stretching and other cosmetic transforms are confined to this layer so that
they can never leak into a GeoTIFF product.

Modules:

* ``stretch`` - percentile-based contrast stretching (pure NumPy functions).
* ``preprocess`` - OpenCV resize and optional Gaussian blur with explicit kernel sizes.
* ``preview`` - RGB, false-colour and single-band PNG composites plus ``hypersat preview``.
"""
