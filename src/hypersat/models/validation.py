"""Models for the pre-flight validation report.

Validation deliberately collects *all* findings instead of aborting on the first one: an
operator preparing a multi-gigabyte orthorectification run should learn about the missing
DEM and the unwritable output directory in the same pass. Each finding keeps the name of
the exception type that a fail-fast caller would raise, so the CLI can still exit with a
specific, documented error.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import computed_field

from hypersat.models.base import StrictModel

__all__ = ["CheckStatus", "ValidationCheck", "ValidationReport"]


class CheckStatus(StrEnum):
    """Outcome of a single validation check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    """Suspicious but not fatal; becomes fatal with ``--strict``."""
    SKIPPED = "skipped"
    """Not applicable, e.g. the DEM check when no DEM was supplied."""


class ValidationCheck(StrictModel):
    """A single named check and what it found."""

    name: str
    status: CheckStatus
    message: str
    hint: str | None = None
    error_type: str | None = None
    """Name of the :class:`hypersat.exceptions.HyperSatError` subclass for a failure."""
    context: dict[str, Any] = {}


class ValidationReport(StrictModel):
    """The full result of a validation run."""

    input_path: Path
    resolved_raster_path: Path | None = None
    checks: list[ValidationCheck] = []
    treat_warnings_as_errors: bool = False

    @property
    def failures(self) -> list[ValidationCheck]:
        """Checks that failed outright."""
        return [check for check in self.checks if check.status is CheckStatus.FAILED]

    @property
    def warnings(self) -> list[ValidationCheck]:
        """Checks that produced a non-fatal warning."""
        return [check for check in self.checks if check.status is CheckStatus.WARNING]

    @property
    def blocking(self) -> list[ValidationCheck]:
        """Checks that make this run invalid, honouring ``treat_warnings_as_errors``."""
        if self.treat_warnings_as_errors:
            return self.failures + self.warnings
        return self.failures

    # Exposed as computed fields so that `--json` output carries the verdict and the
    # tallies, instead of forcing a consumer to re-derive them from the check list.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        """True when nothing blocking was found."""
        return not self.blocking

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> dict[str, int]:
        """Number of checks per status."""
        return self.counts()

    def counts(self) -> dict[str, int]:
        """Number of checks per status, for report headers and JSON output."""
        return {
            status.value: sum(1 for check in self.checks if check.status is status)
            for status in CheckStatus
        }
