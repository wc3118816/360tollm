"""M009 深度估计: DepthMap 数据结构.

承载:
- depth: float32 深度图 (H, W), 单位米. 0 表示无效/未知.
- confidence: float32 置信度图 (H, W), ∈ [0, 1]. 1 = 高置信.
- latency_ms: 推理延迟 (毫秒). AC 要求记录.
- backend: 估计器后端名 (dummy / midas / metric3d / zoe / bifuze).
- width / height: 图像尺寸.
- metadata: 扩展字段 (模型版本 / 设备 / 输入分辨率 等).

与 Frame.depth 兼容: DepthMap.depth 可直接赋给 Frame.depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_DEFAULT_CONF = 1.0
_NDIM_2D = 2  # 深度图必须是 2D


@dataclass(slots=True)
class DepthMap:
    """深度估计结果.

    Attributes:
        depth: float32 (H, W) 深度图, 单位米. 0 = 无效.
        confidence: float32 (H, W) 置信度, ∈ [0, 1].
        latency_ms: 推理延迟 (毫秒).
        backend: 后端名 (dummy / midas / metric3d / zoe / bifuze).
        width: 图像宽度 (像素).
        height: 图像高度 (像素).
        metadata: 扩展字段 (模型版本 / device / 输入分辨率 等).
    """

    depth: np.ndarray
    confidence: np.ndarray
    latency_ms: float = 0.0
    backend: str = "unknown"
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        if self.depth.shape != self.confidence.shape:
            raise ValueError(
                f"depth shape {self.depth.shape} != confidence shape {self.confidence.shape}"
            )
        if self.depth.ndim != _NDIM_2D:
            raise ValueError(f"depth must be 2D (H,W), got {self.depth.shape}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms}")
        # 置信度截断到 [0, 1].
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
        if self.width == 0:
            self.width = self.depth.shape[1]
        if self.height == 0:
            self.height = self.depth.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """(H, W)."""
        return self.depth.shape

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容 dict (depth/confidence 转 list)."""
        return {
            "depth": self.depth.tolist(),
            "confidence": self.confidence.tolist(),
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, Any]:
        """统计摘要 (不包含完整数组, 用于日志/JSON)."""
        valid = self.depth > 0
        n_valid = int(valid.sum())
        n_total = self.depth.size
        return {
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "latency_ms": round(self.latency_ms, 2),
            "valid_pixels": n_valid,
            "total_pixels": n_total,
            "valid_ratio": round(n_valid / n_total, 4) if n_total > 0 else 0.0,
            "depth_min": float(self.depth[valid].min()) if n_valid > 0 else 0.0,
            "depth_max": float(self.depth[valid].max()) if n_valid > 0 else 0.0,
            "depth_mean": float(self.depth[valid].mean()) if n_valid > 0 else 0.0,
            "conf_mean": float(self.confidence.mean()) if n_total > 0 else 0.0,
            "metadata": self.metadata,
        }
