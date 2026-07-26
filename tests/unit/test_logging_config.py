"""Tests for log formatting and logger configuration."""

from __future__ import annotations

import json
import logging

from hypersat.logging_config import (
    LOGGER_NAMESPACE,
    JsonFormatter,
    LogFormat,
    TextFormatter,
    configure_logging,
    get_logger,
)


def _record(**extras: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="hypersat.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="stage completed",
        args=(),
        exc_info=None,
    )
    for key, value in extras.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_one_object_with_extras() -> None:
    payload = json.loads(JsonFormatter().format(_record(stage="orthorectify", duration_s=1.5)))

    assert payload["message"] == "stage completed"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "hypersat.test"
    assert payload["stage"] == "orthorectify"
    assert payload["duration_s"] == 1.5
    assert payload["timestamp"].endswith("+00:00")


def test_text_formatter_appends_sorted_extras() -> None:
    rendered = TextFormatter().format(_record(stage="preview", band=3))

    assert "stage completed" in rendered
    assert rendered.endswith("band=3 stage='preview'")


def test_text_formatter_without_extras_has_no_separator() -> None:
    assert "|" not in TextFormatter().format(_record())


def test_configure_logging_is_idempotent() -> None:
    first = configure_logging(level=logging.DEBUG, log_format=LogFormat.JSON)
    second = configure_logging(level=logging.WARNING, log_format=LogFormat.TEXT)

    assert first is second
    assert len(second.handlers) == 1
    assert second.level == logging.WARNING
    assert not second.propagate
    assert isinstance(second.handlers[0].formatter, TextFormatter)


def test_get_logger_nests_module_names_under_the_package_namespace() -> None:
    assert get_logger("io.reader").name == f"{LOGGER_NAMESPACE}.io.reader"
    assert get_logger("hypersat.cli").name == "hypersat.cli"
    assert get_logger(LOGGER_NAMESPACE).name == LOGGER_NAMESPACE
