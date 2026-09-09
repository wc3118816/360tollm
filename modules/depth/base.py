"""M009 深度估计基类 + 统计.

DepthEstimator ABC:
    estimate(image) -> DepthMap         单帧推理
    estimate_batch(images) -> list[DepthMap]  批量推理 (AC: 可对离线数据批量推理)

DepthStats:
    累计 n_frames / total_latency_ms / avg_latency_ms / p95_latency_ms
    供健康监控与 M044 延迟优化用.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.logging import get_logger

from .types import DepthMap

# p95 至少需要的延迟样本数 (少于此数退化为 avg).
_MIN_LATENCIES_FOR_P95 = 2
# 保留的延迟历史长度上限 (防无限增长).
_MAX_LATENCY_HISTORY = 1000
# 图像维度常量.
_NDIM_GRAYSCALE = 2
_NDIM_COLOR = 3
_CHANNELS_RGB = 3

_LOG = get_logger("modules.depth")


@dataclass
class DepthStats:
    """深度估计器运行统计."""

    n_frames: int = 0
    total_latency_ms: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    backend: str = ""

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.n_frames if self.n_frames > 0 else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if len(self.latencies_ms) < _MIN_LATENCIES_FOR_P95:
            return self.avg_latency_ms
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def record(self, latency_ms: float) -> None:
        self.n_frames += 1
        self.total_latency_ms += latency_ms
        self.latencies_ms.append(latency_ms)
        # 防止无限增长 (保留最近 _MAX_LATENCY_HISTORY 条用于 p95).
        if len(self.latencies_ms) > _MAX_LATENCY_HISTORY:
            self.latencies_ms = self.latencies_ms[-_MAX_LATENCY_HISTORY:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_frames": self.n_frames,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


class DepthEstimator(ABC):
    """深度估计器抽象基类.

    子类必须实现:
        _estimate(image) -> DepthMap    单帧推理 (不含计时, 由 estimate 包装)

    可选重写:
        describe() -> dict              返回后端描述
        warmup() -> None                预热 (加载模型等)
    """

    backend_name: str = "abstract"

    def __init__(self, device: str = "cpu", max_resolution: int = 1024) -> None:
        self.device = device
        self.max_resolution = max_resolution
        self._stats = DepthStats(backend=self.backend_name)
        self._log = _LOG.bind(backend=self.backend_name, device=device)

    def estimate(self, image: np.ndarray) -> DepthMap:
        """单帧深度估计. 自动计时 + 统计.

        Args:
            image: RGB uint8 ndarray (H, W, 3) 或 灰度 (H, W).

        Returns:
            DepthMap.
        """
        img = self._preprocess(image)
        t0 = time.perf_counter()
        result = self._estimate(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result.latency_ms = latency_ms
        result.backend = self.backend_name
        result.metadata.setdefault("device", self.device)
        result.metadata.setdefault("max_resolution", self.max_resolution)
        self._stats.record(latency_ms)
        self._log.debug(
            "depth estimated",
            latency_ms=round(latency_ms, 2),
            shape=result.shape,
        )
        return result

    def estimate_batch(self, images: Sequence[np.ndarray]) -> list[DepthMap]:
        """批量推理 (AC: 可对离线数据批量推理).

        Args:
            images: 图像序列.

        Returns:
            list[DepthMap], 长度 == len(images).
        """
        self._log.info("batch depth estimate start", n_images=len(images))
        results: list[DepthMap] = []
        for i, img in enumerate(images):
            dm = self.estimate(img)
            dm.metadata["batch_index"] = i
            results.append(dm)
        self._log.info(
            "batch depth estimate done",
            n_images=len(images),
            avg_latency_ms=round(self._stats.avg_latency_ms, 2),
        )
        return results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理: 灰度转 RGB + 限制最大分辨率 (下采样)."""
        img = np.asarray(image)
        if img.ndim == _NDIM_GRAYSCALE:  # 灰度 → RGB
            img = np.stack([img, img, img], axis=-1)
        if img.ndim != _NDIM_COLOR or img.shape[2] != _CHANNELS_RGB:
            raise ValueError(f"image must be (H,W,3) or (H,W), got {img.shape}")
        # 下采样到 max_resolution.
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > self.max_resolution:
            scale = self.max_resolution / max_side
            new_h, new_w = int(h * scale), int(w * scale)
            img = _resize_nearest(img, new_h, new_w)
        return img

    @abstractmethod
    def _estimate(self, image: np.ndarray) -> DepthMap:
        """子类实现: 单帧推理 (不含计时). 输入已预处理 (RGB uint8, ≤ max_resolution)."""

    def describe(self) -> dict[str, Any]:
        """后端描述."""
        return {
            "backend": self.backend_name,
            "device": self.device,
            "max_resolution": self.max_resolution,
        }

    def warmup(self) -> None:  # noqa: B027
        """预热 (加载模型 / 首次推理). 默认空实现, 子类可选重写."""

    @property
    def stats(self) -> DepthStats:
        return self._stats


def _resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """最近邻下采样 (无外部依赖)."""
    h, w = img.shape[:2]
    # 简单整数倍下采样用步长采样.
    sy = max(1, h // new_h)
    sx = max(1, w // new_w)
    return img[::sy, ::sx][:new_h, :new_w]
