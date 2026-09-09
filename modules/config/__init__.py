"""统一配置入口。

设计:
- 默认读取 ``configs/default.yaml``.
- 若存在 ``configs/local.yaml`` (gitignored), 自动合并覆盖.
- 若环境变量 ``TOLLM_CONFIG_PATH`` 指向一个 yaml, 它最后覆盖.
- 所有字段都可通过带 ``TOLLM_`` 前缀的环境变量覆盖, 嵌套用 ``__``.
  例: ``TOLLM_LOG__LEVEL=DEBUG`` -> settings.log.level == "DEBUG".
"""

from __future__ import annotations

from .settings import get_settings, reload_settings

__all__ = ["get_settings", "reload_settings"]
