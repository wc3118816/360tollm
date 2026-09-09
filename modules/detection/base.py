"""M019 目标检测基类 + 统计.

Detector ABC:
    detect(image, timestamp) -> DetectionList    单帧推理
    detect_batch(images) -> list[DetectionList]  批量推理

DetectionStats:
    累计 n_frames / total_latency_ms / avg_latency_ms / p95_latency_ms / fps
    供 AC (可在设定 FPS 下稳定运行; 推理延迟可测) 验证.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.logging import get_logger

from .types import DetectionList

# p95 至少需要的延迟样本数.
_MIN_LATENCIES_FOR_P95 = 2
# 保留的延迟历史长度上限.
_MAX_LATENCY_HISTORY = 1000
# 图像维度常量.
_NDIM_GRAYSCALE = 2
_NDIM_COLOR = 3
_CHANNELS_RGB = 3

_LOG = get_logger("modules.detection")


@dataclass
class DetectionStats:
    """目标检测器运行统计."""

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

    @property
    def avg_fps(self) -> float:
        """基于平均延迟的 FPS 估算."""
        if self.avg_latency_ms <= 0:
            return 0.0
        return 1000.0 / self.avg_latency_ms

    def record(self, latency_ms: float) -> None:
        """记录一帧的延迟."""
        self.n_frames += 1
        self.total_latency_ms += latency_ms
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > _MAX_LATENCY_HISTORY:
            self.latencies_ms = self.latencies_ms[-_MAX_LATENCY_HISTORY:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_frames": self.n_frames,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "avg_fps": round(self.avg_fps, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


class Detector(ABC):
    """目标检测器抽象基类.

    子类必须实现:
        _detect(image) -> list[Detection]    单帧推理 (不含计时, 由 detect 包装)

    可选重写:
        describe() -> dict                     返回后端描述
        warmup() -> None                       预热
    """

    backend_name: str = "abstract"

    def __init__(self, device: str = "cpu", max_resolution: int = 1024) -> None:
        self.device = device
        self.max_resolution = max_resolution
        self._stats = DetectionStats(backend=self.backend_name)
        self._log = _LOG.bind(backend=self.backend_name, device=device)

    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> DetectionList:
        """单帧目标检测. 自动计时 + 统计.

        Args:
            image: RGB uint8 (H, W, 3) 或 灰度 (H, W).
            timestamp: 帧时间戳 (秒).

        Returns:
            DetectionList.
        """
        img, img_shape = self._preprocess(image)
        t0 = time.perf_counter()
        detections = self._detect(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._stats.record(latency_ms)
        result = DetectionList(
            detections=detections,
            latency_ms=latency_ms,
            backend=self.backend_name,
            timestamp=timestamp,
            image_shape=img_shape,
            metadata={"device": self.device, "max_resolution": self.max_resolution},
        )
        self._log.debug(
            "detection done",
            latency_ms=round(latency_ms, 2),
            n_detections=len(detections),
            shape=img_shape,
        )
        return result

    def detect_batch(
        self, images: Sequence[np.ndarray], timestamps: Sequence[float] | None = None
    ) -> list[DetectionList]:
        """批量推理.

        Args:
            images: 图像序列.
            timestamps: 时间戳序列 (None 则从 0 递增).

        Returns:
            list[DetectionList], 长度 == len(images).
        """
        if timestamps is None:
            timestamps = [float(i) for i in range(len(images))]
        self._log.info("batch detection start", n_images=len(images))
        results: list[DetectionList] = []
        for i, (img, ts) in enumerate(zip(images, timestamps, strict=True)):
            dl = self.detect(img, timestamp=ts)
            dl.metadata["batch_index"] = i
            results.append(dl)
        self._log.info(
            "batch detection done",
            n_images=len(images),
            avg_latency_ms=round(self._stats.avg_latency_ms, 2),
            avg_fps=round(self._stats.avg_fps, 2),
        )
        return results

    @abstractmethod
    def _detect(self, image: np.ndarray) -> list:
        """子类实现: 单帧推理, 返回 list[Detection]."""

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """预处理 + 返回 (image, (H, W))."""
        img = np.asarray(image)
        if img.ndim not in (_NDIM_GRAYSCALE, _NDIM_COLOR):
            raise ValueError(f"image must be 2D/3D, got {img.ndim}D")
        if img.ndim == _NDIM_COLOR and img.shape[2] != _CHANNELS_RGB:
            raise ValueError(f"image must have 3 channels, got {img.shape[2]}")
        h, w = img.shape[:2]
        return img, (h, w)

    def describe(self) -> dict[str, Any]:
        """后端描述."""
        return {
            "backend": self.backend_name,
            "device": self.device,
            "max_resolution": self.max_resolution,
        }

    def warmup(self) -> None:  # noqa: B027
        """预热 (加载模型等)."""

    @property
    def stats(self) -> DetectionStats:
        return self._stats
