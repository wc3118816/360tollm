"""structlog 包装: 提供结构化、可配置的日志.

设计:
- 默认输出到 stdout, 通过 settings.log 配置级别与格式.
- 支持 ``log_level_override`` 语法 ``"modules.ingest=DEBUG modules.perf=WARN"``
  以便局部调试单个模块.
- ``get_logger(name)`` 是公开 API; 其它模块应只 import 它.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from modules.config import get_settings

_CONFIGURED = False


def _parse_overrides(raw: str) -> dict[str, int]:
    """解析 ``"a.b=DEBUG c=WARN"`` -> ``{"a.b": logging.DEBUG, "c": logging.WARN}``."""
    out: dict[str, int] = {}
    if not raw:
        return out
    for token in raw.split():
        if "=" not in token:
            continue
        name, level = token.split("=", 1)
        name = name.strip()
        try:
            out[name] = getattr(logging, level.strip().upper())
        except AttributeError:
            # 跳过非法级别.
            continue
    return out


def configure_logging() -> None:
    """根据 settings 初始化 structlog. 幂等; 多次调用只配置一次."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    s = get_settings()
    root_level = getattr(logging, s.log.level.upper(), logging.INFO)

    # 标准库 logging 兜底: structlog 内部仍走 logging handlers.
    logging.basicConfig(
        level=root_level,
        stream=sys.stdout,
        format="%(message)s",
    )

    if s.log.format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(root_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # 应用模块级 override.
    for name, level in _parse_overrides(s.log.log_level_override).items():
        logging.getLogger(name).setLevel(level)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """返回命名 logger. 首次调用会自动 configure_logging()."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def reset_logging_for_tests() -> None:
    """测试辅助: 重置配置以便重新读取 settings."""
    global _CONFIGURED
    _CONFIGURED = False
    # structlog 缓存了 logger; 测试期间需要重置.
    structlog.reset_defaults()
