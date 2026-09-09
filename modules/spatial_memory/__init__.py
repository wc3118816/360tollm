"""M031 Spatial Memory 子包.

公开 API:
    ObjectObservation  - 单次观测记录 (timestamp + position + confidence)
    TrackedObject      - 跨帧跟踪物体 (观测历史 + current_position)
    SpatialMemory      - 空间记忆管理器 (observe + where_is + find_recent)
    SpatialMemoryConfig - 配置

用法:
    from modules.spatial_memory import SpatialMemory
    mem = SpatialMemory()
    mem.observe(object_list, pose=pose)
    pos = mem.where_is("T0001")  # AC: 刚才那个物体现在在哪里
"""

from __future__ import annotations

from .memory import SpatialMemory, SpatialMemoryConfig
from .types import ObjectObservation, TrackedObject

__all__ = [
    "ObjectObservation",
    "TrackedObject",
    "SpatialMemory",
    "SpatialMemoryConfig",
]
