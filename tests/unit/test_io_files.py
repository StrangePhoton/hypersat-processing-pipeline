"""Tests for size formatting, checksums and file description."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hypersat.exceptions import RasterReadError
from hypersat.io.files import (
    describe_file,
    directory_size_bytes,
    format_bytes,
    sha256_checksum,
)


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1024**2, "1.0 MiB"),
        (3 * 1024**3, "3.0 GiB"),
        (1024**5, "1.0 PiB"),
    ],
)
def test_format_bytes_uses_binary_prefixes(size_bytes: int, expected: str) -> None:
    assert format_bytes(size_bytes) == expected


def test_format_bytes_rejects_negative_sizes() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        format_bytes(-1)


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hyperspectral bytes" * 1000
    target = tmp_path / "payload.bin"
    target.write_bytes(payload)

    assert sha256_checksum(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_is_chunk_size_independent(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(bytes(range(256)) * 40)

    assert sha256_checksum(target, chunk_size=7) == sha256_checksum(target, chunk_size=4096)


def test_sha256_of_missing_file_is_a_domain_error(tmp_path: Path) -> None:
    with pytest.raises(RasterReadError) as raised:
        sha256_checksum(tmp_path / "absent.bin")

    assert raised.value.exit_code == 5
    assert raised.value.hint is not None


def test_describe_file_reports_size_and_skips_hashing_by_default(tmp_path: Path) -> None:
    target = tmp_path / "scene.bin"
    target.write_bytes(b"x" * 2048)

    info = describe_file(target)

    assert info.size_bytes == 2048
    assert info.size_human == "2.0 KiB"
    assert info.sha256 is None
    assert info.modified_utc is not None
    assert info.modified_utc.endswith("+00:00")


def test_describe_file_hashes_on_request(tmp_path: Path) -> None:
    target = tmp_path / "scene.bin"
    target.write_bytes(b"abc")

    info = describe_file(target, compute_checksum=True)

    assert info.sha256 == hashlib.sha256(b"abc").hexdigest()


def test_describe_file_of_missing_path_is_a_domain_error(tmp_path: Path) -> None:
    with pytest.raises(RasterReadError):
        describe_file(tmp_path / "absent.bin")


def test_directory_size_sums_nested_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 25)

    assert directory_size_bytes(tmp_path) == 125


def test_directory_size_of_empty_directory_is_zero(tmp_path: Path) -> None:
    assert directory_size_bytes(tmp_path / "does-not-exist") == 0
