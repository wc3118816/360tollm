"""M021 2D→3D 目标关联: Object3D 数据结构.

承载:
- object_id: str 唯一标识 (如 "chair_001").
- label: str 类别名 (来自 Detection).
- center: float32 (3,) 世界坐标中心 [x, y, z] (米).
- bbox_3d: float32 (8, 3) 3D 包围盒 8 个顶点 (世界坐标, 可选).
- size_3d: float32 (3,) 3D 尺寸 [dx, dy, dz] (米, 可选).
- confidence: float ∈ [0, 1] (来自 Detection).
- frame_id: int 关联帧 ID.
- timestamp: float 帧时间戳.
- metadata: 扩展字段 (source_bbox_2d / depth_value / pose_backend 等).

Object3DList:
- objects: list[Object3D]
- frame_id / timestamp / n_objects
- summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_NDIM_3D = 3
_BBOX_3D_VERTS = 8
_SIZE_DIM = 3
_DEFAULT_CONF = 1.0


@dataclass(slots=True)
class Object3D:
    """3D 空间中的物体.

    Attributes:
        object_id: 唯一标识.
        label: 类别名.
        center: float32 (3,) 世界坐标中心 [x, y, z] (米).
        bbox_3d: float32 (8, 3) 3D 包围盒 8 顶点 (世界坐标, 可选).
        size_3d: float32 (3,) 尺寸 [dx, dy, dz] (米, 可选).
        confidence: 置信度 ∈ [0, 1].
        frame_id: 关联帧 ID.
        timestamp: 帧时间戳.
        metadata: 扩展字段.
    """

    object_id: str
    label: str
    center: np.ndarray
    bbox_3d: np.ndarray = field(
        default_factory=lambda: np.zeros((_BBOX_3D_VERTS, _NDIM_3D), dtype=np.float32)
    )
    size_3d: np.ndarray = field(default_factory=lambda: np.zeros(_SIZE_DIM, dtype=np.float32))
    confidence: float = _DEFAULT_CONF
    frame_id: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32).ravel()
        if self.center.shape[0] != _NDIM_3D:
            raise ValueError(f"center must be (3,), got {self.center.shape}")
        self.bbox_3d = np.asarray(self.bbox_3d, dtype=np.float32)
        if self.bbox_3d.size == 0:
            self.bbox_3d = self.bbox_3d.reshape(_BBOX_3D_VERTS, _NDIM_3D)
        if self.bbox_3d.shape != (_BBOX_3D_VERTS, _NDIM_3D):
            raise ValueError(f"bbox_3d must be (8,3), got {self.bbox_3d.shape}")
        self.size_3d = np.asarray(self.size_3d, dtype=np.float32).ravel()
        if self.size_3d.shape[0] != _SIZE_DIM:
            raise ValueError(f"size_3d must be (3,), got {self.size_3d.shape}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be ∈ [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict (JSON 兼容)."""
        return {
            "object_id": self.object_id,
            "class": self.label,
            "center": self.center.tolist(),
            "bbox_3d": self.bbox_3d.tolist(),
            "size_3d": self.size_3d.tolist(),
            "confidence": round(float(self.confidence), 4),
            "frame_id": int(self.frame_id),
            "timestamp": float(self.timestamp),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Object3DList:
    """单帧 3D 物体列表.

    Attributes:
        objects: list[Object3D].
        frame_id: 帧 ID.
        timestamp: 帧时间戳.
        metadata: 扩展字段.
    """

    objects: list[Object3D] = field(default_factory=list)
    frame_id: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def labels(self) -> list[str]:
        return list({o.label for o in self.objects})

    def filter_by_label(self, label: str) -> list[Object3D]:
        return [o for o in self.objects if o.label == label]

    def filter_by_confidence(self, threshold: float = 0.5) -> list[Object3D]:
        return [o for o in self.objects if o.confidence >= threshold]

    def summary(self) -> dict[str, Any]:
        confs = [o.confidence for o in self.objects]
        return {
            "n_objects": self.n_objects,
            "labels": self.labels,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_objects": self.n_objects,
            "objects": [o.to_dict() for o in self.objects],
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
