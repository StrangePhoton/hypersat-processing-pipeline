"""Choosing bands by wavelength instead of by band number.

A band index is a property of one sensor's layout: band 42 is a different part of the
spectrum on EnMAP than on PRISMA or Sentinel-2, and it moves between product versions. A
centre wavelength is a physical quantity, so selection here is always expressed in
nanometres and resolved against the wavelengths the product itself reports (extracted by
:mod:`hypersat.io.inspect`).

Selection is nearest-neighbour with a tolerance. The tolerance exists because "nearest"
alone is dangerous: asked for 1600 nm on a VNIR-only product, an untolerant search would
happily return the 1000 nm edge band and the caller would never know the request was
unsatisfiable. Ties resolve to the lower band index, which matters on instruments whose
detectors overlap - EnMAP's VNIR reaches about 1000 nm and its SWIR starts around 900 nm,
so near 950 nm two bands can be almost equidistant from the target. Every match reports
its distance so callers can surface a poor-but-legal match rather than hide it.

These are pure functions over a sequence of wavelengths; nothing here opens a raster.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hypersat.exceptions import InvalidWavelengthMetadataError

__all__ = [
    "BLUE_NM",
    "DEFAULT_TOLERANCE_NM",
    "GREEN_NM",
    "NIR_NM",
    "RED_NM",
    "BandMatch",
    "false_colour_bands",
    "nearest_band",
    "nearest_bands",
    "true_colour_bands",
]

DEFAULT_TOLERANCE_NM = 20.0
"""How far a band may sit from the requested wavelength before the match is rejected.

20 nm is narrower than a typical multispectral band and wider than a hyperspectral
sampling interval (EnMAP samples at roughly 5-12 nm), so a hyperspectral product always
has a candidate while a coarse product fails loudly instead of returning something far
away.
"""

BLUE_NM = 490.0
GREEN_NM = 560.0
RED_NM = 665.0
NIR_NM = 842.0
"""Generic broadband centres used for composites.

These are conventional visible/near-infrared centres, close to Sentinel-2's B2/B3/B4/B8.
They are *targets* to search for, not band numbers, so they work on any instrument whose
wavelengths cover them - which is the entire point of selecting this way.
"""

_COMPOSITE_TOLERANCE_NM = 30.0
"""Composites tolerate a slightly wider miss than analysis, since previews are cosmetic."""


@dataclass(frozen=True, slots=True)
class BandMatch:
    """The band chosen for a requested wavelength.

    Attributes:
        index: One-based band index, ready to pass to the reader.
        wavelength_nm: Centre wavelength the product reports for that band.
        target_nm: Wavelength that was requested.
        distance_nm: Absolute difference between the two; zero for an exact hit.
    """

    index: int
    wavelength_nm: float
    target_nm: float
    distance_nm: float


def _usable_wavelengths(wavelengths_nm: Sequence[float | None]) -> list[tuple[int, float]]:
    """Return ``(one-based index, wavelength)`` pairs for bands with usable metadata.

    Bands whose wavelength is missing, non-finite or non-positive are skipped rather than
    guessed at: a partially annotated product should still be usable through the bands it
    does describe.

    Raises:
        InvalidWavelengthMetadataError: If no band has a usable wavelength.
    """
    usable = [
        (position + 1, float(value))
        for position, value in enumerate(wavelengths_nm)
        if value is not None and math.isfinite(value) and value > 0.0
    ]
    if not usable:
        raise InvalidWavelengthMetadataError(
            "No band has a usable centre wavelength, so bands cannot be selected by wavelength.",
            hint="Run `hypersat inspect` to see what wavelength metadata the product "
            "carries. Selecting explicit band indices is the alternative when a product "
            "genuinely has none.",
            context={"band_count": len(wavelengths_nm)},
        )
    return usable


def nearest_band(
    wavelengths_nm: Sequence[float | None],
    target_nm: float,
    *,
    tolerance_nm: float = DEFAULT_TOLERANCE_NM,
) -> BandMatch:
    """Find the band whose centre wavelength is closest to ``target_nm``.

    Args:
        wavelengths_nm: Centre wavelengths per band, in band order; entries may be
            ``None`` where the product does not report one.
        target_nm: Wavelength to search for, in nanometres.
        tolerance_nm: Largest acceptable distance from the target.

    Returns:
        The matching band, including how far it sits from the target.

    Raises:
        ValueError: If ``target_nm`` or ``tolerance_nm`` is not a positive finite number.
        InvalidWavelengthMetadataError: If no band has usable wavelength metadata, or the
            closest band lies outside the tolerance.
    """
    if not math.isfinite(target_nm) or target_nm <= 0.0:
        raise ValueError(f"target_nm must be a positive finite wavelength, got {target_nm}")
    if not math.isfinite(tolerance_nm) or tolerance_nm < 0.0:
        raise ValueError(f"tolerance_nm must be a non-negative finite value, got {tolerance_nm}")

    usable = _usable_wavelengths(wavelengths_nm)
    # `min` keeps the first minimum it meets, and the pairs are in ascending band order,
    # so an exact tie resolves to the lower index.
    index, wavelength = min(usable, key=lambda pair: abs(pair[1] - target_nm))
    distance = abs(wavelength - target_nm)

    if distance > tolerance_nm:
        raise InvalidWavelengthMetadataError(
            f"No band lies within {tolerance_nm:g} nm of {target_nm:g} nm; the closest is "
            f"band {index} at {wavelength:g} nm.",
            hint="Widen tolerance_nm if that band is acceptable, or check that the "
            "product actually covers this part of the spectrum.",
            context={
                "target_nm": target_nm,
                "tolerance_nm": tolerance_nm,
                "closest_band": index,
                "closest_wavelength_nm": wavelength,
                "distance_nm": distance,
            },
        )
    return BandMatch(
        index=index,
        wavelength_nm=wavelength,
        target_nm=target_nm,
        distance_nm=distance,
    )


def nearest_bands(
    wavelengths_nm: Sequence[float | None],
    targets_nm: Sequence[float],
    *,
    tolerance_nm: float = DEFAULT_TOLERANCE_NM,
) -> tuple[BandMatch, ...]:
    """Resolve several wavelengths at once, preserving the order requested.

    The same band may be returned more than once when two targets fall inside one band,
    which is legitimate: a coarse product genuinely observes both with the same detector.

    Args:
        wavelengths_nm: Centre wavelengths per band, in band order.
        targets_nm: Wavelengths to search for, in the order the caller wants them.
        tolerance_nm: Largest acceptable distance from each target.

    Returns:
        One match per target, in the same order.

    Raises:
        ValueError: If ``targets_nm`` is empty or contains an invalid wavelength.
        InvalidWavelengthMetadataError: If any target cannot be satisfied.
    """
    if not targets_nm:
        raise ValueError("targets_nm must contain at least one wavelength")
    return tuple(
        nearest_band(wavelengths_nm, target, tolerance_nm=tolerance_nm) for target in targets_nm
    )


def true_colour_bands(
    wavelengths_nm: Sequence[float | None],
    *,
    tolerance_nm: float = _COMPOSITE_TOLERANCE_NM,
) -> tuple[BandMatch, BandMatch, BandMatch]:
    """Select red, green and blue bands for a natural-looking composite.

    Args:
        wavelengths_nm: Centre wavelengths per band, in band order.
        tolerance_nm: Largest acceptable distance from each nominal centre.

    Returns:
        Matches for red, green and blue, in that order - the order an RGB image expects.

    Raises:
        InvalidWavelengthMetadataError: If the product does not cover the visible range.
    """
    red, green, blue = nearest_bands(
        wavelengths_nm,
        (RED_NM, GREEN_NM, BLUE_NM),
        tolerance_nm=tolerance_nm,
    )
    return red, green, blue


def false_colour_bands(
    wavelengths_nm: Sequence[float | None],
    *,
    tolerance_nm: float = _COMPOSITE_TOLERANCE_NM,
) -> tuple[BandMatch, BandMatch, BandMatch]:
    """Select near-infrared, red and green bands for the classic false-colour composite.

    Mapping near-infrared into the red channel makes vegetation, which reflects strongly
    there, stand out - the conventional infrared rendering, not an analysis product.

    Args:
        wavelengths_nm: Centre wavelengths per band, in band order.
        tolerance_nm: Largest acceptable distance from each nominal centre.

    Returns:
        Matches for near-infrared, red and green, in that order.

    Raises:
        InvalidWavelengthMetadataError: If the product does not reach the near infrared.
    """
    nir, red, green = nearest_bands(
        wavelengths_nm,
        (NIR_NM, RED_NM, GREEN_NM),
        tolerance_nm=tolerance_nm,
    )
    return nir, red, green
