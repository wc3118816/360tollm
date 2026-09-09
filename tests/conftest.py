"""pytest 全局 fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# 让测试期间使用一份稳定的离线配置, 避免污染真实环境.
os.environ.setdefault("TOLLM_CONFIG_PATH", "")
os.environ.setdefault("TOLLM_LOG__LEVEL", "DEBUG")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """根据测试文件路径自动打 unit / integration 标记.

    tests/unit/**    -> unit
    tests/integration/** -> integration
    其它 (含 tests/*.py) -> unit
    """
    for item in items:
        path = Path(item.fspath).resolve()
        parts = path.parts
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def project_root() -> Path:
    from modules.config.loader import project_root

    return project_root()


@pytest.fixture
def fresh_settings():
    """每个测试函数独立的 settings, 避免单例缓存干扰."""
    from modules.config import reload_settings

    return reload_settings()


@pytest.fixture(autouse=True)
def reset_logging():
    from modules.logging import reset_logging_for_tests

    reset_logging_for_tests()
    yield
    reset_logging_for_tests()
