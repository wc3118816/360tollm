"""modules.config.settings 单测."""

from __future__ import annotations

from modules.config import get_settings, reload_settings


def test_default_settings_load_yaml(fresh_settings) -> None:
    s = fresh_settings
    assert s.project.name == "tollm"
    assert s.project.timezone == "Asia/Shanghai"
    assert s.ingest.source == "offline"
    assert s.world_model.units == "meters"


def test_env_var_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("TOLLM_LOG__LEVEL", "DEBUG")
    monkeypatch.setenv("TOLLM_INGEST__TARGET_FPS", "10.0")
    s = reload_settings()
    assert s.log.level == "DEBUG"
    assert s.ingest.target_fps == 10.0


def test_llm_api_key_env_lookup(monkeypatch) -> None:
    monkeypatch.setenv("TOLLM_LLM__API_KEY_ENV", "TOLLM_TEST_KEY")
    monkeypatch.setenv("TOLLM_TEST_KEY", "secret-123")
    from modules.config.settings import get_llm_api_key

    s = reload_settings()
    assert s.llm.api_key_env == "TOLLM_TEST_KEY"
    assert get_llm_api_key() == "secret-123"


def test_settings_singleton_cached() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
