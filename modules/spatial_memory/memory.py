"""M031 Spatial Memory: 跨帧空间记忆管理器.

SpatialMemory:
- observe(object_list, pose, timestamp) → 添加新观测 + 跨帧关联.
- get_object(tracked_id) → TrackedObject.
- where_is(tracked_id) → "刚才那个物体现在在哪里" (AC).
- find_recent(label, time_window) → 查询最近观测的物体.
- all_objects() → list[TrackedObject].
- cleanup(max_age) → 移除过老的物体.

跨帧关联算法 (简化版):
- 新观测物体与已有 TrackedObject 的 current_position 距离 < association_threshold → 关联.
- 同 label + 距离近 → 同一 tracked_id.
- 否则创建新 TrackedObject.

用法:
    mem = SpatialMemory(SpatialMemoryConfig(association_threshold=1.0))
    mem.observe(object_list, pose, timestamp=1.0)
    pos = mem.where_is("T0001")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.logging import get_logger
from modules.objects import Object3D, Object3DList
from modules.vo.types import Pose

from .types import ObjectObservation, TrackedObject

_LOG = get_logger("modules.spatial_memory")

_NDIM_3D = 3
# 默认关联阈值 (米): 距离 < 1.0 → 同一物体.
_DEFAULT_ASSOC_THRESHOLD = 1.0
# 默认最大记忆时长 (秒): 超过 60s 未观测 → 清除.
_DEFAULT_MAX_AGE = 60.0
# tracked_id 前缀.
_TRACKED_ID_PREFIX = "T"


@dataclass
class SpatialMemoryConfig:
    """空间记忆配置."""

    # 跨帧关联距离阈值 (米).
    association_threshold: float = _DEFAULT_ASSOC_THRESHOLD
    # 最大记忆时长 (秒).
    max_age: float = _DEFAULT_MAX_AGE
    # 是否启用 cleanup.
    auto_cleanup: bool = True


@dataclass
class SpatialMemory:
    """跨帧空间记忆.

    管理所有 TrackedObject + 查询 API.

    Attributes:
        objects: dict[str, TrackedObject] tracked_id → TrackedObject.
        cfg: 配置.
        _id_counter: tracked_id 自增计数器.
    """

    cfg: SpatialMemoryConfig = field(default_factory=SpatialMemoryConfig)
    objects: dict[str, TrackedObject] = field(default_factory=dict)
    _id_counter: int = 0

    def __post_init__(self) -> None:
        self._log = _LOG

    def observe(
        self,
        object_list: Object3DList,
        pose: Pose | None = None,
    ) -> list[str]:
        """添加新观测 + 跨帧关联.

        Args:
            object_list: M021 输出的 3D 物体列表.
            pose: 相机位姿 (可选, 用于元数据).

        Returns:
            list[str] 新增/关联的 tracked_id 列表.
        """
        timestamp = object_list.timestamp
        frame_id = object_list.frame_id
        tracked_ids: list[str] = []

        for obj in object_list.objects:
            tracked_id = self._associate_object(obj, timestamp, frame_id)
            tracked_ids.append(tracked_id)

        # 自动清理.
        if self.cfg.auto_cleanup and timestamp > 0:
            self.cleanup(timestamp)

        self._log.debug(
            "observe done",
            n_new_objects=len(object_list.objects),
            n_total=len(self.objects),
            frame_id=frame_id,
        )
        return tracked_ids

    def _associate_object(self, obj: Object3D, timestamp: float, frame_id: int) -> str:
        """将单个 Object3D 关联到 TrackedObject (或新建)."""
        # 1. 找最近的同 label TrackedObject.
        best_id: str | None = None
        best_dist = float("inf")
        for tid, tracked in self.objects.items():
            if tracked.label != obj.label:
                continue
            if tracked.n_observations == 0:
                continue
            dist = float(np.linalg.norm(tracked.current_position - obj.center))
            if dist < self.cfg.association_threshold and dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None:
            # 2a. 关联到已有 TrackedObject.
            tracked = self.objects[best_id]
            obs = ObjectObservation(
                timestamp=timestamp,
                frame_id=frame_id,
                position=obj.center.copy(),
                confidence=obj.confidence,
                source_object_id=obj.object_id,
            )
            tracked.add_observation(obs)
            return best_id

        # 2b. 创建新 TrackedObject.
        tracked_id = self._next_id()
        tracked = TrackedObject(
            tracked_id=tracked_id,
            label=obj.label,
            metadata={"first_frame_id": frame_id, "first_timestamp": timestamp},
        )
        obs = ObjectObservation(
            timestamp=timestamp,
            frame_id=frame_id,
            position=obj.center.copy(),
            confidence=obj.confidence,
            source_object_id=obj.object_id,
        )
        tracked.add_observation(obs)
        self.objects[tracked_id] = tracked
        return tracked_id

    def _next_id(self) -> str:
        """生成下一个 tracked_id."""
        self._id_counter += 1
        return f"{_TRACKED_ID_PREFIX}{self._id_counter:04d}"

    # === 查询 API ===

    def get_object(self, tracked_id: str) -> TrackedObject | None:
        """获取 TrackedObject."""
        return self.objects.get(tracked_id)

    def where_is(self, tracked_id: str) -> np.ndarray | None:
        """AC: "刚才那个物体现在在哪里".

        Returns:
            float32 (3,) 世界坐标 或 None (不存在).
        """
        tracked = self.objects.get(tracked_id)
        if tracked is None or tracked.n_observations == 0:
            return None
        return tracked.current_position.copy()

    def find_recent(
        self,
        label: str | None = None,
        time_window: float = 10.0,
        current_time: float = 0.0,
    ) -> list[TrackedObject]:
        """查询最近观测的物体.

        Args:
            label: 限定类别 (None 不限).
            time_window: 时间窗口 (秒).
            current_time: 当前时间.

        Returns:
            list[TrackedObject] 在时间窗口内被观测的物体.
        """
        threshold = current_time - time_window
        results = [
            obj
            for obj in self.objects.values()
            if obj.last_seen >= threshold and (label is None or obj.label == label)
        ]
        # 按最后观测时间降序.
        results.sort(key=lambda o: o.last_seen, reverse=True)
        return results

    def find_by_label(self, label: str) -> list[TrackedObject]:
        """按类别查询所有物体."""
        return [o for o in self.objects.values() if o.label == label]

    def find_nearest_to_position(
        self, position: np.ndarray, label: str | None = None
    ) -> tuple[str, float] | None:
        """找离指定位置最近的物体.

        Returns:
            (tracked_id, distance) 或 None.
        """
        pos = np.asarray(position, dtype=np.float64)
        nearest_id: str | None = None
        nearest_dist = float("inf")
        for tid, tracked in self.objects.items():
            if label is not None and tracked.label != label:
                continue
            if tracked.n_observations == 0:
                continue
            d = float(np.linalg.norm(tracked.current_position - pos))
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = tid
        if nearest_id is None:
            return None
        return (nearest_id, nearest_dist)

    def all_objects(self) -> list[TrackedObject]:
        """获取所有 TrackedObject."""
        return list(self.objects.values())

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def labels(self) -> list[str]:
        return list({o.label for o in self.objects.values()})

    # === 管理 ===

    def cleanup(self, current_time: float) -> int:
        """清除超时未观测的物体.

        Args:
            current_time: 当前时间.

        Returns:
            int 清除数量.
        """
        threshold = current_time - self.cfg.max_age
        removed = [tid for tid, tracked in self.objects.items() if tracked.last_seen < threshold]
        for tid in removed:
            del self.objects[tid]
        if removed:
            self._log.debug("cleanup", n_removed=len(removed), n_remaining=len(self.objects))
        return len(removed)

    def clear(self) -> None:
        """清空所有记忆."""
        self.objects.clear()
        self._id_counter = 0

    def summary(self) -> dict[str, Any]:
        """模型摘要."""
        return {
            "n_objects": self.n_objects,
            "labels": self.labels,
            "association_threshold": self.cfg.association_threshold,
            "max_age": self.cfg.max_age,
            "n_total_observations": sum(o.n_observations for o in self.objects.values()),
        }

    def serialize(self) -> str:
        """序列化为 JSON 字符串."""
        import json

        data = {
            "objects": [o.to_dict() for o in self.objects.values()],
            "config": {
                "association_threshold": self.cfg.association_threshold,
                "max_age": self.cfg.max_age,
            },
            "id_counter": self._id_counter,
            "summary": self.summary(),
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
