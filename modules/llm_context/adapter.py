"""M037 LLM 上下文适配器: World Model → LLM 最小必要上下文.

LLMContextAdapter:
- adapt(query, world_model, spatial_memory) → ContextSnippet.
- 按 query_type 检索相关物体 (不发送整张地图).
- 几何事实优先 (位置/距离/方向).
- 保留置信度 + 时间信息.

支持的查询类型:
- WHERE_IS: "X 在哪里?" → 返回位置事实.
- DISTANCE: "X 离 Y 多远?" → 返回距离事实.
- DIRECTION: "X 在 Y 的哪个方向?" → 返回方向事实.
- COUNT: "有几个 X?" → 返回数量事实.
- LIST_OBJECTS: "有哪些物体?" → 返回物体列表 (受 max_objects 限制).
- NEAREST: "离我最近的 X?" → 返回最近物体事实.
- RECENT: "刚才那个物体?" → 从 SpatialMemory 查询时间窗口内物体.

AC: 同一问题不会无条件发送整张地图.
实现: max_objects 限制 + 按 query_type 过滤 → 只返回相关物体.

用法:
    adapter = LLMContextAdapter()
    query = ContextQuery(question="椅子在哪里?", query_type=QueryType.WHERE_IS, target_label="chair")
    snippet = adapter.adapt(query, world_model, spatial_memory)
    print(snippet.text)  # LLM 可读文本.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.logging import get_logger
from modules.spatial_memory import SpatialMemory
from modules.world_model import WorldModel

from .types import ContextQuery, ContextSnippet, QueryType

_LOG = get_logger("modules.llm_context")

_NDIM_3D = 3
# 方向阈值 (用于描述 "左/右/前/后").
_DIRECTION_EPS = 1e-6
# 两物体 ID 分隔符.
_ID_SEPARATOR = ":"
# 两物体 ID 最小段数.
_MIN_ID_PARTS = 2


@dataclass
class LLMContextAdapter:
    """LLM 上下文适配器.

    将 WorldModel + SpatialMemory 转换为 LLM 可读的最小上下文.

    Attributes:
        max_objects_default: 默认最大物体数.
    """

    max_objects_default: int = 10

    def __post_init__(self) -> None:
        self._log = _LOG

    def adapt(
        self,
        query: ContextQuery,
        world_model: WorldModel | None = None,
        spatial_memory: SpatialMemory | None = None,
    ) -> ContextSnippet:
        """根据查询生成 LLM 上下文.

        Args:
            query: 上下文查询.
            world_model: WorldModel (M029).
            spatial_memory: SpatialMemory (M031, 可选).

        Returns:
            ContextSnippet LLM 可读上下文.
        """
        n_total = 0
        if world_model is not None:
            n_total = world_model.n_objects

        # 按查询类型生成上下文.
        if query.query_type == QueryType.WHERE_IS:
            snippet = self._adapt_where_is(query, world_model)
        elif query.query_type == QueryType.DISTANCE:
            snippet = self._adapt_distance(query, world_model)
        elif query.query_type == QueryType.DIRECTION:
            snippet = self._adapt_direction(query, world_model)
        elif query.query_type == QueryType.COUNT:
            snippet = self._adapt_count(query, world_model)
        elif query.query_type == QueryType.LIST_OBJECTS:
            snippet = self._adapt_list_objects(query, world_model)
        elif query.query_type == QueryType.NEAREST:
            snippet = self._adapt_nearest(query, world_model)
        elif query.query_type == QueryType.RECENT:
            snippet = self._adapt_recent(query, spatial_memory)
        else:
            snippet = self._adapt_custom(query, world_model)

        snippet.n_objects_total = n_total
        self._log.debug(
            "adapt",
            query_type=query.query_type.value,
            n_objects=snippet.n_objects_filtered,
            n_total=n_total,
        )
        return snippet

    def _adapt_where_is(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """WHERE_IS: "X 在哪里?" → 位置事实."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)

        # 按 label 过滤.
        objects = self._filter_by_label(model, query.target_label)
        objects = self._limit_objects(objects, query.max_objects)

        if not objects:
            label = query.target_label or "物体"
            return ContextSnippet(
                text=f"未找到{label}。",
                n_objects_filtered=0,
            )

        facts: list[str] = []
        obj_summaries: list[dict[str, Any]] = []
        for obj in objects:
            pos = obj.center
            fact = (
                f"{obj.label}({obj.object_id}) 在 "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) 米处，"
                f"置信度 {obj.confidence:.2f}。"
            )
            facts.append(fact)
            obj_summaries.append(
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "position": pos.tolist(),
                    "confidence": round(float(obj.confidence), 4),
                }
            )

        text = "\n".join(facts)
        avg_conf = float(np.mean([o.confidence for o in objects]))
        return ContextSnippet(
            text=text,
            facts=facts,
            objects=obj_summaries,
            confidence=avg_conf,
            n_objects_filtered=len(objects),
        )

    def _adapt_distance(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """DISTANCE: "X 离 Y 多远?" → 距离事实."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)

        # 需要两个物体 ID.
        if query.target_object_id is None:
            return self._distance_to_reference(query, model)

        # 两个物体间距离.
        ids = query.target_object_id.split(_ID_SEPARATOR)
        if len(ids) < _MIN_ID_PARTS:
            return ContextSnippet(text="需要两个物体 ID (格式: a:b)。")
        obj_a = model.get_object(ids[0])
        obj_b = model.get_object(ids[1])
        if obj_a is None or obj_b is None:
            return ContextSnippet(text="未找到目标物体。")
        dist = model.object_distance(ids[0], ids[1])
        if dist is None:
            return ContextSnippet(text="无法计算距离。")
        fact = (
            f"{obj_a.label}({ids[0]}) 距 {obj_b.label}({ids[1]}) {dist:.2f} 米，"
            f"置信度 {min(obj_a.confidence, obj_b.confidence):.2f}。"
        )
        return ContextSnippet(
            text=fact,
            facts=[fact],
            objects=[
                {
                    "object_a": ids[0],
                    "object_b": ids[1],
                    "distance": round(dist, 4),
                }
            ],
            confidence=min(obj_a.confidence, obj_b.confidence),
            n_objects_filtered=2,
        )

    def _distance_to_reference(self, query: ContextQuery, model: WorldModel) -> ContextSnippet:
        """物体到参考位置的距离."""
        if query.reference_position is None:
            return ContextSnippet(text="需要指定两个物体或参考位置。")
        obj = self._find_first_by_label(model, query.target_label)
        if obj is None:
            return ContextSnippet(text="未找到目标物体。")
        ref = query.reference_position
        dist = float(np.linalg.norm(obj.center - ref))
        fact = (
            f"{obj.label}({obj.object_id}) 距参考位置 {dist:.2f} 米，置信度 {obj.confidence:.2f}。"
        )
        return ContextSnippet(
            text=fact,
            facts=[fact],
            objects=[
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "distance": round(dist, 4),
                    "confidence": round(float(obj.confidence), 4),
                }
            ],
            confidence=obj.confidence,
            n_objects_filtered=1,
        )

    def _adapt_direction(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """DIRECTION: "X 在 Y 的哪个方向?" → 方向事实."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)

        if query.reference_position is not None:
            return self._direction_to_reference(query, model)

        # 两物体间方向.
        if query.target_object_id:
            ids = query.target_object_id.split(_ID_SEPARATOR)
            if len(ids) >= _MIN_ID_PARTS:
                return self._direction_between_objects(model, ids)

        return ContextSnippet(text="需要参考位置或两个物体 ID。")

    def _direction_to_reference(self, query: ContextQuery, model: WorldModel) -> ContextSnippet:
        """物体相对参考位置的方向."""
        obj = self._find_first_by_label(model, query.target_label)
        if obj is None:
            return ContextSnippet(text="未找到目标物体。")
        ref = query.reference_position
        assert ref is not None
        direction = obj.center.astype(np.float64) - ref
        dir_text = self._direction_to_text(direction)
        fact = f"{obj.label}({obj.object_id}) 在参考位置的{dir_text}，置信度 {obj.confidence:.2f}。"
        return ContextSnippet(
            text=fact,
            facts=[fact],
            n_objects_filtered=1,
        )

    def _direction_between_objects(self, model: WorldModel, ids: list[str]) -> ContextSnippet:
        """两物体间方向."""
        obj_a = model.get_object(ids[0])
        obj_b = model.get_object(ids[1])
        if obj_a is None or obj_b is None:
            return ContextSnippet(text="未找到目标物体。")
        direction = model.object_direction(ids[0], ids[1])
        if direction is None:
            return ContextSnippet(text="无法计算方向。")
        dir_text = self._direction_to_text(direction.astype(np.float64))
        fact = f"{obj_b.label}({ids[1]}) 在 {obj_a.label}({ids[0]}) 的{dir_text}。"
        return ContextSnippet(
            text=fact,
            facts=[fact],
            n_objects_filtered=2,
        )

    def _adapt_count(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """COUNT: "有几个 X?" → 数量事实."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)

        if query.target_label:
            count = len(model.filter_by_label(query.target_label))
            fact = f"有 {count} 个{query.target_label}。"
        else:
            count = model.n_objects
            fact = f"共有 {count} 个物体。"
        return ContextSnippet(
            text=fact,
            facts=[fact],
            confidence=1.0,
            n_objects_filtered=0,  # 数量查询不返回物体详情.
        )

    def _adapt_list_objects(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """LIST_OBJECTS: "有哪些物体?" → 物体列表 (受 max_objects 限制)."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)

        objects = self._filter_by_label(model, query.target_label)
        n_before = len(objects)
        objects = self._limit_objects(objects, query.max_objects)

        if not objects:
            return ContextSnippet(
                text="未找到物体。",
                n_objects_filtered=0,
            )

        facts: list[str] = [f"{o.label}({o.object_id})" for o in objects]
        text = f"找到 {n_before} 个物体" + (
            f"（显示前 {len(objects)} 个）" if len(objects) < n_before else ""
        )
        text += "：" + ", ".join(facts) + "。"
        return ContextSnippet(
            text=text,
            facts=facts,
            objects=[
                {
                    "object_id": o.object_id,
                    "label": o.label,
                    "confidence": round(float(o.confidence), 4),
                }
                for o in objects
            ],
            confidence=1.0,
            n_objects_filtered=len(objects),
        )

    def _adapt_nearest(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """NEAREST: "离我最近的 X?" → 最近物体事实."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)

        ref = query.reference_position
        if ref is None:
            ref = np.zeros(_NDIM_3D, dtype=np.float64)

        # 手动找最近物体 (WorldModel 没有 find_nearest_to_position).
        objects = self._filter_by_label(model, query.target_label)
        if not objects:
            return ContextSnippet(text="未找到物体。")

        nearest_obj = min(objects, key=lambda o: float(np.linalg.norm(o.center - ref)))
        dist = float(np.linalg.norm(nearest_obj.center - ref))
        fact = (
            f"最近的{nearest_obj.label}({nearest_obj.object_id}) 在 {dist:.2f} 米处，"
            f"置信度 {nearest_obj.confidence:.2f}。"
        )
        return ContextSnippet(
            text=fact,
            facts=[fact],
            objects=[
                {
                    "object_id": nearest_obj.object_id,
                    "label": nearest_obj.label,
                    "distance": round(dist, 4),
                    "confidence": round(float(nearest_obj.confidence), 4),
                }
            ],
            confidence=nearest_obj.confidence,
            n_objects_filtered=1,
        )

    def _adapt_recent(self, query: ContextQuery, mem: SpatialMemory | None) -> ContextSnippet:
        """RECENT: "刚才那个物体?" → 从 SpatialMemory 查询."""
        if mem is None:
            return ContextSnippet(text="无空间记忆。", n_objects_filtered=0)

        time_window = query.time_window or 10.0
        # 用最后观测时间作为 current_time.
        current_time = max((o.last_seen for o in mem.objects.values()), default=0.0)
        recent = mem.find_recent(
            label=query.target_label,
            time_window=time_window,
            current_time=current_time,
        )
        recent = self._limit_objects(recent, query.max_objects)

        if not recent:
            return ContextSnippet(text="近期无物体观测。", n_objects_filtered=0)

        facts: list[str] = []
        obj_summaries: list[dict[str, Any]] = []
        for tracked in recent:
            pos = tracked.current_position
            fact = (
                f"{tracked.label}({tracked.tracked_id}) 最后在 "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) 米处被观测，"
                f"时间 {tracked.last_seen:.1f} 秒前，"
                f"置信度 {tracked.current_confidence:.2f}。"
            )
            facts.append(fact)
            obj_summaries.append(
                {
                    "tracked_id": tracked.tracked_id,
                    "label": tracked.label,
                    "position": pos.tolist(),
                    "last_seen": round(float(tracked.last_seen), 4),
                    "confidence": round(float(tracked.current_confidence), 4),
                }
            )

        text = "\n".join(facts)
        avg_conf = float(np.mean([t.current_confidence for t in recent]))
        return ContextSnippet(
            text=text,
            facts=facts,
            objects=obj_summaries,
            confidence=avg_conf,
            n_objects_filtered=len(recent),
        )

    def _adapt_custom(self, query: ContextQuery, model: WorldModel | None) -> ContextSnippet:
        """CUSTOM: 自定义查询 → 返回简要摘要."""
        if model is None:
            return ContextSnippet(text="无法获取世界模型。", n_objects_filtered=0)
        summary = model.summary()
        text = f"场景包含 {summary['n_objects']} 个物体，类别: {', '.join(summary['labels'])}。"
        return ContextSnippet(
            text=text,
            facts=[text],
            confidence=1.0,
            n_objects_filtered=0,
        )

    # === 辅助方法 ===

    def _filter_by_label(self, model: WorldModel, label: str | None) -> list[Any]:
        """按 label 过滤物体."""
        if label is None:
            return list(model.objects.values())
        return model.filter_by_label(label)

    def _limit_objects(self, objects: list[Any], max_n: int) -> list[Any]:
        """限制物体数量 (AC: 不发送整张地图)."""
        if len(objects) <= max_n:
            return objects
        return objects[:max_n]

    def _find_first_by_label(self, model: WorldModel, label: str | None) -> Any | None:
        """按 label 找第一个物体."""
        objs = self._filter_by_label(model, label)
        return objs[0] if objs else None

    def _direction_to_text(self, direction: np.ndarray) -> str:
        """方向向量 → 自然语言描述.

        约定 (world 坐标系):
        - +x: 前/北
        - +y: 上
        - +z: 右/东
        """
        x, y, z = direction
        parts: list[str] = []
        # 前后.
        if abs(x) > _DIRECTION_EPS:
            parts.append("前方" if x > 0 else "后方")
        # 上下.
        if abs(y) > _DIRECTION_EPS:
            parts.append("上方" if y > 0 else "下方")
        # 左右.
        if abs(z) > _DIRECTION_EPS:
            parts.append("右侧" if z > 0 else "左侧")
        if not parts:
            return "同一位置"
        return "的".join(parts)
