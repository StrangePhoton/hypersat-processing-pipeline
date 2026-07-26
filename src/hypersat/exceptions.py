"""Exception hierarchy for the HyperSat pipeline.

Every error raised deliberately by this package derives from :class:`HyperSatError`.
Errors carry three pieces of information:

* ``message`` - what went wrong, in domain language;
* ``hint`` - what the operator can do about it (optional but strongly encouraged);
* ``context`` - structured key/value details for logs and JSON reports.

Each exception class also declares an :attr:`HyperSatError.exit_code`, which the CLI
uses as the process exit status. Exit codes are part of the public contract because
the pipeline is expected to be driven by shell scripts and CI jobs.

Exit code map:

===== ==========================================================================
Code  Meaning
===== ==========================================================================
0     Success
1     Unexpected internal error (bug)
2     CLI usage error (raised by Typer/Click, not by this module)
3     Configuration error
4     Product / input validation error
5     Raster I/O error
6     Processing error
7     Pipeline orchestration error
8     Missing optional dependency
9     Requested functionality is not implemented yet
===== ==========================================================================
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = [
    "ConfigurationError",
    "DEMError",
    "DependencyError",
    "GDALBindingsUnavailableError",
    "HyperSatError",
    "InvalidWavelengthMetadataError",
    "MissingDEMError",
    "MissingGeoreferencingError",
    "MissingRPCMetadataError",
    "NotImplementedYetError",
    "OrthorectificationError",
    "OutputPathError",
    "PipelineError",
    "ProcessingError",
    "ProductStructureError",
    "ProductValidationError",
    "QualityMaskError",
    "RasterIOError",
    "RasterMetadataError",
    "RasterReadError",
    "RasterWriteError",
    "ReprojectionError",
    "SpectralAnalysisError",
    "StageExecutionError",
    "UnreadableDEMError",
]


class HyperSatError(Exception):
    """Base class for all deliberate HyperSat failures.

    Args:
        message: Human-readable description of the failure.
        hint: Optional actionable remediation advice shown to the operator.
        context: Optional structured details (paths, counts, CRS strings, ...).
    """

    exit_code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.context:
            details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
            parts.append(f"[{details}]")
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for quality-control reports."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "hint": self.hint,
            "context": self.context,
            "exit_code": self.exit_code,
        }


class ConfigurationError(HyperSatError):
    """The pipeline configuration file or CLI options are invalid or inconsistent."""

    exit_code: ClassVar[int] = 3


class ProductValidationError(HyperSatError):
    """The input product failed a pre-flight validation check."""

    exit_code: ClassVar[int] = 4


class ProductStructureError(ProductValidationError):
    """A required file or directory of the satellite product is absent."""


class RasterMetadataError(ProductValidationError):
    """The raster opens, but its metadata is missing or unusable for the request."""


class MissingGeoreferencingError(RasterMetadataError):
    """The raster carries neither a CRS/affine transform nor an alternative sensor model."""


class MissingRPCMetadataError(RasterMetadataError):
    """Orthorectification was requested, but the raster has no RPC sensor model.

    Raised instead of silently degrading to a plain reprojection, which would
    produce a geometrically incorrect product (see ``docs/orthorectification.md``).
    """


class InvalidWavelengthMetadataError(RasterMetadataError):
    """Per-band centre wavelengths are absent, unparsable or not monotonic."""


class DEMError(ProductValidationError):
    """Base class for digital-elevation-model problems."""


class MissingDEMError(DEMError):
    """The configured DEM path does not exist."""


class UnreadableDEMError(DEMError):
    """The DEM exists but cannot be opened, or lacks a usable vertical reference."""


class OutputPathError(ProductValidationError):
    """The output directory cannot be created or is not writable."""


class RasterIOError(HyperSatError):
    """Base class for raster read/write failures."""

    exit_code: ClassVar[int] = 5


class RasterReadError(RasterIOError):
    """A raster or one of its windows/bands could not be read."""


class RasterWriteError(RasterIOError):
    """A raster product could not be written to its destination."""


class ProcessingError(HyperSatError):
    """Base class for failures inside a processing algorithm."""

    exit_code: ClassVar[int] = 6


class OrthorectificationError(ProcessingError):
    """RPC/DEM-based orthorectification failed or could not be performed correctly."""


class ReprojectionError(ProcessingError):
    """CRS reprojection, grid alignment or resampling failed."""


class QualityMaskError(ProcessingError):
    """Quality-mask generation failed or produced an inconsistent class map."""


class SpectralAnalysisError(ProcessingError):
    """A spectral index, profile extraction or statistics computation failed."""


class PipelineError(HyperSatError):
    """Base class for orchestration failures."""

    exit_code: ClassVar[int] = 7


class StageExecutionError(PipelineError):
    """A pipeline stage failed; wraps the underlying domain error."""


class DependencyError(HyperSatError):
    """An optional third-party dependency is required but unavailable."""

    exit_code: ClassVar[int] = 8


class GDALBindingsUnavailableError(DependencyError):
    """The ``osgeo.gdal`` Python bindings are needed but not importable."""


class NotImplementedYetError(HyperSatError):
    """A CLI command or stage exists as a contract but has no implementation yet.

    Used only while the roadmap in ``docs/roadmap.md`` is being worked through, so that
    the CLI surface can be reviewed before the algorithms land.
    """

    exit_code: ClassVar[int] = 9
