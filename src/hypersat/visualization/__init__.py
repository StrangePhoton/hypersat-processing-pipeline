"""Preview rendering. Never writes scientific rasters, only 8-bit PNG previews.

Percentile stretching and other cosmetic transforms are confined to this layer so that
they can never leak into a GeoTIFF product.

Planned modules (milestone 4):

* ``stretch`` - percentile-based contrast stretching (pure functions).
* ``preview`` - RGB, false-colour and single-band PNG composites.
"""
