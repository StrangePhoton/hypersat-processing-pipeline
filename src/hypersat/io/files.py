"""Filesystem utilities: sizes, timestamps and checksums.

Checksums exist for the quality-control report: a run must be able to state exactly which
bytes it produced. Hashing is therefore always opt-in, because a hyperspectral cube is
large enough that hashing it is a measurable cost rather than a free convenience.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

from hypersat.exceptions import RasterReadError
from hypersat.models.product import FileInfo

__all__ = [
    "CHUNK_SIZE_BYTES",
    "derive_product_id",
    "describe_file",
    "directory_size_bytes",
    "format_bytes",
    "sha256_checksum",
]

CHUNK_SIZE_BYTES = 1 << 20
"""Read granularity for hashing: 1 MiB keeps memory flat on multi-gigabyte products."""

_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
_UNIT_STEP = 1024.0
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def derive_product_id(path: Path) -> str:
    """Derive a filesystem-safe product id from an input path.

    Args:
        path: Raster file or product directory.

    Returns:
        A lower-case ASCII slug; falls back to ``"product"`` if nothing usable remains.
    """
    # Use the stem for anything that looks like a file (has a suffix), including paths
    # that do not exist yet. ``is_file()`` would be wrong for a not-yet-checked path and
    # would leave the extension in the slug.
    if path.exists():
        token = path.stem if path.is_file() else path.name
    else:
        token = path.stem if path.suffix else path.name
    slug = _SLUG_PATTERN.sub("_", token.lower()).strip("_")
    return slug or "product"


def format_bytes(size_bytes: int) -> str:
    """Render a byte count using binary prefixes.

    Args:
        size_bytes: Non-negative size in bytes.

    Returns:
        A short human-readable string such as ``"1.4 GiB"``.

    Raises:
        ValueError: If ``size_bytes`` is negative.
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must not be negative")
    size = float(size_bytes)
    for unit in _UNITS:
        if size < _UNIT_STEP or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= _UNIT_STEP
    raise AssertionError("unreachable")  # pragma: no cover


def directory_size_bytes(directory: Path) -> int:
    """Return the total size of every regular file under ``directory``.

    Symbolic links are not followed, so a link loop cannot inflate the total or hang the
    walk.

    Args:
        directory: Directory to measure.

    Returns:
        Total size in bytes; ``0`` for an empty or non-existent directory.
    """
    total = 0
    for entry in directory.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


def sha256_checksum(path: Path, chunk_size: int = CHUNK_SIZE_BYTES) -> str:
    """Return the SHA-256 hex digest of a file, read in fixed-size chunks.

    Args:
        path: File to hash.
        chunk_size: Read granularity in bytes.

    Returns:
        Lowercase hexadecimal digest.

    Raises:
        RasterReadError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise RasterReadError(
            "Could not read file to compute its checksum.",
            hint="Check that the path exists and the process has read permission.",
            context={"path": str(path), "reason": str(error)},
        ) from error
    return digest.hexdigest()


def describe_file(path: Path, *, compute_checksum: bool = False) -> FileInfo:
    """Collect size, modification time and optionally the SHA-256 digest of a file.

    Args:
        path: File to describe.
        compute_checksum: Whether to hash the file. Off by default.

    Returns:
        A populated :class:`hypersat.models.product.FileInfo`.

    Raises:
        RasterReadError: If the file's metadata cannot be read.
    """
    try:
        stat_result = path.stat()
    except OSError as error:
        raise RasterReadError(
            "Could not read file metadata.",
            hint="Verify the path exists and is accessible to this process.",
            context={"path": str(path), "reason": str(error)},
        ) from error

    modified = dt.datetime.fromtimestamp(stat_result.st_mtime, tz=dt.UTC).isoformat()
    return FileInfo(
        path=path,
        size_bytes=stat_result.st_size,
        size_human=format_bytes(stat_result.st_size),
        modified_utc=modified,
        sha256=sha256_checksum(path) if compute_checksum else None,
    )
