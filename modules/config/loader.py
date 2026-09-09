"""YAML 配置文件加载与合并工具。

只负责读 YAML -> dict, 不做 pydantic 校验.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    # pyproject.toml is at the repo root; this file is modules/config/loader.py.
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: three levels up from this file.
    return here.parents[2]


def project_root() -> Path:
    """返回工程根目录 (包含 pyproject.toml 的目录)."""
    return _project_root()


def resolve_path(path_str: str) -> Path:
    """把配置里的相对路径解析为相对工程根目录的绝对路径."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并; override 中非 dict 值直接覆盖 base 同名键."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_stack() -> dict[str, Any]:
    """按以下顺序合并 yaml 配置 (后者覆盖前者):

    1. configs/default.yaml        (committed)
    2. configs/local.yaml          (gitignored, optional)
    3. $TOLLM_CONFIG_PATH          (optional, explicit)
    """
    root = project_root()
    merged: dict[str, Any] = {}

    default_path = root / "configs" / "default.yaml"
    merged = deep_merge(merged, load_yaml(default_path))

    local_path = root / "configs" / "local.yaml"
    if local_path.exists():
        merged = deep_merge(merged, load_yaml(local_path))

    env_path_str = os.environ.get("TOLLM_CONFIG_PATH")
    if env_path_str:
        env_path = Path(env_path_str).expanduser().resolve()
        merged = deep_merge(merged, load_yaml(env_path))

    return merged
