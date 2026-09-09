"""M031 Spatial Memory: 跨帧空间记忆数据结构.

ObjectObservation:
- timestamp: float 观测时间戳 (秒).
- frame_id: int 帧 ID.
- position: float32 (3,) 世界坐标 [x, y, z] (米).
- confidence: float ∈ [0, 1].
- source_object_id: str M021 生成的 object_id.

TrackedObject:
- tracked_id: str 跨帧唯一标识 (如 "T0001").
- label: str 类别名.
- observations: list[ObjectObservation] 观测历史 (按时间排序).
- current_position: float32 (3,) 最新位置.
- current_confidence: float 最新置信度.
- first_seen: float 首次观测时间.
- last_seen: float 最后观测时间.
- n_observations: int 观测次数.
- metadata: 扩展字段.

查询:
- position_at(timestamp) → 该时间最近的位置.
- trajectory() → 所有位置 (N, 3).
- is_still(threshold) → 是否静止 (位置变化 < threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_NDIM_3D = 3
_DEFAULT_CONF = 1.0
_STILL_THRESHOLD = 0.1  # 静止判断阈值 (米).
_MIN_OBS_FOR_MOTION = 2  # 判断运动至少需要的观测数.


@dataclass(slots=True)
class ObjectObservation:
    """单次物体观测记录.

    Attributes:
        timestamp: 观测时间戳 (秒).
        frame_id: 帧 ID.
        position: float32 (3,) 世界坐标 (米).
        confidence: 置信度 ∈ [0, 1].
        source_object_id: M021 生成的 object_id.
    """

    timestamp: float
    frame_id: int
    position: np.ndarray
    confidence: float = _DEFAULT_CONF
    source_object_id: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).ravel()
        if self.position.shape[0] != _NDIM_3D:
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be ∈ [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "frame_id": int(self.frame_id),
            "position": self.position.tolist(),
            "confidence": round(float(self.confidence), 4),
            "source_object_id": self.source_object_id,
        }


@dataclass
class TrackedObject:
    """跨帧跟踪的物体.

    Attributes:
        tracked_id: 跨帧唯一标识.
        label: 类别名.
        observations: 观测历史 (按时间排序).
        metadata: 扩展字段.
    """

    tracked_id: str
    label: str
    observations: list[ObjectObservation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def first_seen(self) -> float:
        if not self.observations:
            return 0.0
        return self.observations[0].timestamp

    @property
    def last_seen(self) -> float:
        if not self.observations:
            return 0.0
        return self.observations[-1].timestamp

    @property
    def current_position(self) -> np.ndarray:
        if not self.observations:
            return np.zeros(_NDIM_3D, dtype=np.float32)
        return self.observations[-1].position.copy()

    @property
    def current_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return self.observations[-1].confidence

    @property
    def first_frame_id(self) -> int:
        if not self.observations:
            return 0
        return self.observations[0].frame_id

    @property
    def last_frame_id(self) -> int:
        if not self.observations:
            return 0
        return self.observations[-1].frame_id

    def add_observation(self, obs: ObjectObservation) -> None:
        """添加观测 (自动按时间排序)."""
        self.observations.append(obs)
        self.observations.sort(key=lambda o: o.timestamp)

    def position_at(self, timestamp: float) -> np.ndarray | None:
        """获取指定时间最近的位置.

        Returns:
            float32 (3,) 或 None (无观测).
        """
        if not self.observations:
            return None
        # 找时间最近的观测.
        nearest = min(self.observations, key=lambda o: abs(o.timestamp - timestamp))
        return nearest.position.copy()

    def trajectory(self) -> np.ndarray:
        """获取所有位置 (N, 3)."""
        if not self.observations:
            return np.zeros((0, _NDIM_3D), dtype=np.float32)
        return np.array([o.position for o in self.observations], dtype=np.float32)

    def is_still(self, threshold: float = _STILL_THRESHOLD) -> bool:
        """判断物体是否静止 (位置变化 < threshold)."""
        if len(self.observations) < _MIN_OBS_FOR_MOTION:
            return True  # 单次观测视为静止.
        traj = self.trajectory()
        diffs = np.diff(traj, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return bool(np.all(distances < threshold))

    def total_distance(self) -> float:
        """总移动距离 (米)."""
        if len(self.observations) < _MIN_OBS_FOR_MOTION:
            return 0.0
        traj = self.trajectory()
        diffs = np.diff(traj, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracked_id": self.tracked_id,
            "label": self.label,
            "n_observations": self.n_observations,
            "first_seen": float(self.first_seen),
            "last_seen": float(self.last_seen),
            "current_position": self.current_position.tolist(),
            "current_confidence": round(float(self.current_confidence), 4),
            "is_still": self.is_still(),
            "total_distance": round(self.total_distance(), 4),
            "metadata": self.metadata,
            "observations": [o.to_dict() for o in self.observations],
        }
