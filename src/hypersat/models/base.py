"""Shared Pydantic base classes.

Every model in this package is strict and immutable:

* ``extra="forbid"`` turns a mistyped configuration key into an error instead of a
  silently ignored setting;
* ``frozen=True`` keeps inspection results and configuration free of hidden mutation;
* ``ser_json_inf_nan="null"`` guarantees that JSON output is *valid* JSON. Raster
  metadata really does contain NaN (a NaN NoData value is common), and the default
  Pydantic behaviour would emit the bare token ``NaN``, which many JSON parsers reject.
  Models that need to preserve the distinction expose an explicit boolean flag instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["StrictModel"]


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields and serialises to valid JSON."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        ser_json_inf_nan="null",
    )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary (``Path`` as string, NaN/Inf as ``null``)."""
        return self.model_dump(mode="json")
