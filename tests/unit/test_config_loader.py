"""modules.config.loader 单测."""

from __future__ import annotations

from pathlib import Path

from modules.config.loader import deep_merge, load_yaml, resolve_path


def test_project_root_contains_pyproject(project_root: Path) -> None:
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "configs" / "default.yaml").exists()


def test_resolve_relative_path_is_absolute(project_root: Path) -> None:
    p = resolve_path("datasets")
    assert p.is_absolute()
    assert p == (project_root / "datasets").resolve()


def test_resolve_absolute_path_passthrough(tmp_path: Path) -> None:
    abs_path = tmp_path / "foo.bin"
    assert resolve_path(str(abs_path)) == abs_path


def test_load_yaml_returns_dict(project_root: Path) -> None:
    data = load_yaml(project_root / "configs" / "default.yaml")
    assert isinstance(data, dict)
    assert data["project"]["name"] == "tollm"
    assert data["ingest"]["source"] == "offline"


def test_load_yaml_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_yaml(tmp_path / "does_not_exist.yaml") == {}


def test_deep_merge_nested() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 20, "e": 30}, "f": 40}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 20, "e": 30}, "d": 3, "f": 40}
    # 不修改原 base.
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_deep_merge_override_non_dict_replaces() -> None:
    base = {"a": {"b": 1}}
    override = {"a": "string-overrides-everything"}
    assert deep_merge(base, override) == {"a": "string-overrides-everything"}
