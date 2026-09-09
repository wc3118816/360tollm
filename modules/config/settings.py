"""Pydantic Settings: 把 yaml + 环境变量映射成强类型对象.

使用:
    from modules.config import get_settings
    settings = get_settings()
    settings.log.level           # "INFO"
    settings.ingest.target_fps   # 2.0

优先级 (高 -> 低):
    显式构造参数 > 环境变量 (TOLLM_ 前缀, __ 嵌套) > yaml 配置栈 > 字段默认值
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .loader import load_config_stack


class ProjectCfg(BaseModel):
    name: str = "tollm"
    version: str = "0.0.1"
    timezone: str = "Asia/Shanghai"


class PathsCfg(BaseModel):
    datasets: str = "datasets"
    recordings: str = "datasets/recordings"
    checkpoints: str = "checkpoints"
    logs: str = "logs"
    cache: str = ".cache"


class LogCfg(BaseModel):
    level: str = "INFO"
    format: Literal["console", "json"] = "console"
    log_level_override: str = ""


class IngestCfg(BaseModel):
    source: Literal["offline", "rtmp", "file"] = "offline"
    rtmp_url: str = ""
    reconnect_attempts: int = 5
    reconnect_backoff_sec: float = 2.0
    target_fps: float = 2.0
    io_timeout_sec: float = 5.0
    low_latency: bool = True


class FrameCfg(BaseModel):
    timestamp_source: Literal["ingest", "decode"] = "ingest"
    drop_stale: bool = True


class CalibrationCfg(BaseModel):
    """M006 相机标定配置.

    AC: "误差阈值可配置" — max_reproj_error.
    """

    max_reproj_error: float = 1.0  # 像素, > 该阈值视为标定失败
    chessboard_cols: int = 9  # 棋盘格内角点列数
    chessboard_rows: int = 6  # 棋盘格内角点行数
    square_size: float = 0.025  # 格子边长 (米)
    min_valid_images: int = 5  # 最少有效图像数


class DepthCfg(BaseModel):
    """M009 深度估计配置.

    AC: 可对离线数据批量推理 → estimate_batch.
    backend 可选: none | dummy | midas | midas_small | metric3d | zoe | bifuze.
    """

    backend: str = "none"
    device: Literal["cpu", "cuda"] = "cpu"
    max_resolution: int = 1024
    near: float = 0.5  # 深度映射下限 (米)
    far: float = 20.0  # 深度映射上限 (米)


class SLAMCfg(BaseModel):
    """M011 Visual Odometry / M013 SLAM 配置.

    backend 可选: none | dummy | orb | feature | feature_vo (M011)
                  | orbslam3 | msvc (M013, 未实现)
    """

    backend: str = "none"
    vocab_path: str = ""
    device: Literal["cpu", "cuda"] = "cpu"
    n_features: int = 1000  # ORB 特征数 (FeatureVO)
    scale_factor: float = 0.1  # 单目尺度因子 (米, 启发式)


class DetectionCfg(BaseModel):
    backend: str = "none"
    device: Literal["cpu", "cuda"] = "cpu"


class SegmentationCfg(BaseModel):
    backend: str = "none"


class PerceptionCfg(BaseModel):
    detection: DetectionCfg = Field(default_factory=DetectionCfg)
    segmentation: SegmentationCfg = Field(default_factory=SegmentationCfg)


class WorldModelCfg(BaseModel):
    coordinate_system: Literal["world", "camera"] = "world"
    units: Literal["meters"] = "meters"
    temporal_window_sec: int = 30
    memory_persist: bool = True


class LLMCfg(BaseModel):
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024


class YamlConfigSource(PydanticBaseSettingsSource):
    """把 ``load_config_stack()`` 的 dict 作为最低优先级 source.

    pydantic-settings v2 中, 显式构造参数 > 环境变量 > yaml source > 默认值.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = load_config_stack()

    def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class Settings(BaseSettings):
    """顶层 Settings.

    优先级 (高 -> 低):
        显式构造参数 > 环境变量 (TOLLM_ 前缀, __ 嵌套) > yaml 配置栈 > 字段默认值.
    """

    model_config = SettingsConfigDict(
        env_prefix="TOLLM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectCfg = Field(default_factory=ProjectCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    log: LogCfg = Field(default_factory=LogCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    frame: FrameCfg = Field(default_factory=FrameCfg)
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    depth: DepthCfg = Field(default_factory=DepthCfg)
    slam: SLAMCfg = Field(default_factory=SLAMCfg)
    perception: PerceptionCfg = Field(default_factory=PerceptionCfg)
    world_model: WorldModelCfg = Field(default_factory=WorldModelCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        secret_settings: PydanticBaseSettingsSource | None = None,
        **kwargs: Any,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # pydantic-settings 2.15+ 把 secret_settings 改名为 file_secret_settings,
        # kwargs 兜底以兼容. 顺序即优先级, 前者覆盖后者.
        extra = []
        # 收集未来版本的额外 secret-like source (保持低优先级).
        for v in kwargs.values():
            if isinstance(v, PydanticBaseSettingsSource):
                extra.append(v)
        sources = [
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSource(settings_cls),
        ]
        if secret_settings is not None:
            sources.append(secret_settings)
        sources.extend(extra)
        return tuple(sources)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局单例 Settings."""
    return Settings()


def reload_settings() -> Settings:
    """清除缓存并重新读取配置 (测试或运行时切换配置后调用)."""
    get_settings.cache_clear()
    return get_settings()


# 便于环境变量驱动的 API key 查询.
def get_llm_api_key() -> str | None:
    s = get_settings()
    return os.environ.get(s.llm.api_key_env)
