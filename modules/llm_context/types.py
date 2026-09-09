"""M037 LLM 上下文适配器: 查询类型 + 上下文片段.

ContextQuery:
- question: str 用户问题.
- query_type: QueryType 枚举 (where_is / distance / direction / count / list_objects / nearest / recent / custom).
- target_label: str | None 目标类别.
- target_object_id: str | None 目标物体 ID.
- reference_position: np.ndarray | None 参考位置 (如 "我左边").
- time_window: float | None 时间窗口 (如 "刚才").
- max_objects: int 最多返回物体数 (防止发送整张地图).

ContextSnippet:
- text: str LLM 可读文本.
- facts: list[str] 几何事实列表.
- objects: list[dict] 相关物体摘要.
- confidence: float 上下文置信度.
- n_objects_filtered: int 被过滤掉的物体数.

原则:
- 几何事实优先 (不发送原始点云).
- 保留置信度 + 时间信息.
- max_objects 限制 → AC: 不无条件发送整张地图.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

_NDIM_3D = 3
# 默认最大物体数 (防止发送整张地图).
_DEFAULT_MAX_OBJECTS = 10


class QueryType(StrEnum):
    """查询类型枚举."""

    WHERE_IS = "where_is"  # "X 在哪里?"
    DISTANCE = "distance"  # "X 离 Y 多远?"
    DIRECTION = "direction"  # "X 在 Y 的哪个方向?"
    COUNT = "count"  # "有几个 X?"
    LIST_OBJECTS = "list_objects"  # "有哪些物体?"
    NEAREST = "nearest"  # "离我最近的 X 是?"
    RECENT = "recent"  # "刚才那个物体?"
    CUSTOM = "custom"  # 自定义


@dataclass
class ContextQuery:
    """LLM 上下文查询.

    Attributes:
        question: 用户原始问题.
        query_type: 查询类型.
        target_label: 目标类别 (如 "chair").
        target_object_id: 目标物体 ID.
        reference_position: 参考位置 (3,) 世界坐标.
        time_window: 时间窗口 (秒), 如 "刚才" = 10.0.
        max_objects: 最多返回物体数.
    """

    question: str
    query_type: QueryType = QueryType.CUSTOM
    target_label: str | None = None
    target_object_id: str | None = None
    reference_position: np.ndarray | None = None
    time_window: float | None = None
    max_objects: int = _DEFAULT_MAX_OBJECTS

    def __post_init__(self) -> None:
        self.max_objects = max(self.max_objects, 1)
        if self.reference_position is not None:
            self.reference_position = np.asarray(self.reference_position, dtype=np.float64).ravel()
            if self.reference_position.shape[0] != _NDIM_3D:
                raise ValueError(
                    f"reference_position must be (3,), got {self.reference_position.shape}"
                )


@dataclass
class ContextSnippet:
    """LLM 上下文片段.

    Attributes:
        text: LLM 可读文本.
        facts: 几何事实列表 (如 "椅子在 (1.0, 2.0, 3.0) 米处").
        objects: 相关物体摘要列表.
        confidence: 上下文置信度.
        n_objects_filtered: 被过滤掉的物体数.
        n_objects_total: 总物体数.
    """

    text: str
    facts: list[str] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    n_objects_filtered: int = 0
    n_objects_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "facts": self.facts,
            "objects": self.objects,
            "confidence": round(float(self.confidence), 4),
            "n_objects_filtered": int(self.n_objects_filtered),
            "n_objects_total": int(self.n_objects_total),
        }
