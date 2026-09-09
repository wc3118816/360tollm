"""M019 目标检测: Detection 数据结构.

承载:
- bbox: int32 (4,) = [x_min, y_min, x_max, y_max] 像素坐标.
- label: 类别名 (str, 如 "person" / "chair" / "cup").
- label_id: 类别 ID (int, 0-based).
- confidence: float ∈ [0, 1].
- timestamp: 帧时间戳 (秒).
- latency_ms: 推理延迟 (毫秒).
- metadata: 扩展字段 (frame_id / backend / image_shape 等).

DetectionList:
- detections: list[Detection]
- latency_ms: 整帧推理延迟
- backend: 检测器后端名
- image_shape: (H, W)
- summary(): 统计
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# bbox 维度.
_BBOX_SIZE = 4
# 默认置信度.
_DEFAULT_CONF = 1.0
# image_shape 维度.
_SHAPE_DIM = 2


@dataclass(slots=True)
class Detection:
    """单个目标检测结果.

    Attributes:
        bbox: int32 (4,) = [x_min, y_min, x_max, y_max] 像素坐标.
        label: 类别名.
        label_id: 类别 ID (0-based).
        confidence: 置信度 ∈ [0, 1].
        metadata: 扩展字段.
    """

    bbox: np.ndarray
    label: str
    label_id: int = 0
    confidence: float = _DEFAULT_CONF
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.int32).ravel()
        if self.bbox.shape[0] != _BBOX_SIZE:
            raise ValueError(f"bbox must be (4,), got {self.bbox.shape}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be ∈ [0,1], got {self.confidence}")
        # bbox 有效性: x_max > x_min, y_max > y_min.
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError(f"invalid bbox {self.bbox.tolist()}: x_max<=x_min or y_max<=y_min")

    @property
    def width(self) -> int:
        """bbox 宽度 (像素)."""
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        """bbox 高度 (像素)."""
        return int(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        """bbox 面积 (像素²)."""
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        """bbox 中心 (cx, cy)."""
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict."""
        return {
            "bbox": self.bbox.tolist(),
            "label": self.label,
            "label_id": int(self.label_id),
            "confidence": round(float(self.confidence), 4),
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center": self.center.tolist(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DetectionList:
    """单帧目标检测结果集合.

    Attributes:
        detections: list[Detection].
        latency_ms: 整帧推理延迟 (毫秒).
        backend: 检测器后端名 (dummy / yoloe / rtdetr).
        timestamp: 帧时间戳 (秒).
        image_shape: (H, W) 原图尺寸.
        metadata: 扩展字段.
    """

    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = "unknown"
    timestamp: float = 0.0
    image_shape: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms}")
        if len(self.image_shape) != _SHAPE_DIM:
            raise ValueError(f"image_shape must be (H,W), got {self.image_shape}")

    @property
    def n_detections(self) -> int:
        return len(self.detections)

    @property
    def labels(self) -> list[str]:
        """所有检测到的类别名 (去重)."""
        return list({d.label for d in self.detections})

    @property
    def fps(self) -> float:
        """基于 latency_ms 的 FPS 估算."""
        if self.latency_ms <= 0:
            return 0.0
        return 1000.0 / self.latency_ms

    def filter_by_confidence(self, threshold: float = 0.5) -> list[Detection]:
        """按置信度过滤."""
        return [d for d in self.detections if d.confidence >= threshold]

    def filter_by_label(self, label: str) -> list[Detection]:
        """按类别名过滤."""
        return [d for d in self.detections if d.label == label]

    def summary(self) -> dict[str, Any]:
        """统计摘要."""
        confs = [d.confidence for d in self.detections]
        return {
            "n_detections": self.n_detections,
            "labels": self.labels,
            "latency_ms": round(self.latency_ms, 2),
            "fps": round(self.fps, 2),
            "backend": self.backend,
            "image_shape": list(self.image_shape),
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict."""
        return {
            "n_detections": self.n_detections,
            "detections": [d.to_dict() for d in self.detections],
            "latency_ms": round(self.latency_ms, 2),
            "fps": round(self.fps, 2),
            "backend": self.backend,
            "timestamp": self.timestamp,
            "image_shape": list(self.image_shape),
            "metadata": self.metadata,
        }
