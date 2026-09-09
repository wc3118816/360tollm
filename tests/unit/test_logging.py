"""modules.logging 单测."""

from __future__ import annotations

import logging

from modules.logging import _parse_overrides, get_logger, reset_logging_for_tests


def test_parse_overrides_basic() -> None:
    out = _parse_overrides("modules.ingest=DEBUG perf=WARN")
    assert out == {"modules.ingest": logging.DEBUG, "perf": logging.WARN}


def test_parse_overrides_skips_invalid_level() -> None:
    out = _parse_overrides("a=NOTALEVEL b=INFO")
    assert out == {"b": logging.INFO}


def test_parse_overrides_empty_returns_empty() -> None:
    assert _parse_overrides("") == {}
    assert _parse_overrides("   ") == {}


def test_get_logger_returns_bound_logger() -> None:
    reset_logging_for_tests()
    log = get_logger("test.unit")
    # Bound logger 应能调用任意级别方法而不抛.
    log.info("hello", component="tollm", value=42)
    log.debug("debug-msg")


def test_logging_idempotent_configure() -> None:
    from modules.logging import configure_logging

    reset_logging_for_tests()
    configure_logging()
    configure_logging()  # 第二次调用应无副作用.
    log = get_logger("test.idempotent")
    log.info("ok")
