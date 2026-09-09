"""M026 Scene Graph: 从 Object3DList 构建场景图 + 空间关系计算.

关系计算 (简化版, M027 的前置):
- near: 距离 < near_threshold (米).
- far: 距离 > far_threshold (米).
- left_of / right_of / in_front_of / behind:
  以 source 节点为参考, target 的相对位置.
- on: target 在 source 上方且距离近.
- inside: target 在 source 的 bbox 内.
- next_to: 距离 < next_to_threshold 且不在上方.
- connected_to: 需外部标注 (默认不计算).

用法:
    builder = SceneGraphBuilder(GraphBuilderConfig())
    graph = builder.build(object_list)
    json_str = graph.serialize()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.logging import get_logger
from modules.objects import Object3D, Object3DList

from .types import RELATIONS, SceneEdge, SceneGraph, SceneNode

_LOG = get_logger("modules.scene_graph.builder")

_NDIM_3D = 3
# 默认距离阈值 (米).
_DEFAULT_NEAR_THRESHOLD = 2.0
_DEFAULT_FAR_THRESHOLD = 5.0
_DEFAULT_NEXT_TO_THRESHOLD = 1.5
_DEFAULT_ON_THRESHOLD = 0.5  # z 方向差 < 此值视为 "on".
_MIN_PAIRS = 2  # 至少 2 个物体才能计算关系.
_SAME_POS_EPS = 1e-6  # 同位置判断阈值.
_AXIS_X = 0
_AXIS_Y = 1
_AXIS_Z = 2


@dataclass
class GraphBuilderConfig:
    """场景图构建配置."""

    near_threshold: float = _DEFAULT_NEAR_THRESHOLD
    far_threshold: float = _DEFAULT_FAR_THRESHOLD
    next_to_threshold: float = _DEFAULT_NEXT_TO_THRESHOLD
    on_threshold: float = _DEFAULT_ON_THRESHOLD
    # 是否计算方向关系 (left/right/front/behind).
    compute_directions: bool = True
    # 是否计算 near/far.
    compute_distance_relations: bool = True
    # 视角方向 (计算 left/right/front/behind 时的参考轴).
    # "x": x 轴为前方; "y": y 轴为前方.
    forward_axis: str = "y"


class SceneGraphBuilder:
    """从 Object3DList 构建场景图.

    用法:
        builder = SceneGraphBuilder()
        graph = builder.build(object_list)
    """

    def __init__(self, cfg: GraphBuilderConfig | None = None) -> None:
        self.cfg = cfg or GraphBuilderConfig()
        self._log = _LOG

    def build(
        self,
        object_list: Object3DList,
        existing_graph: SceneGraph | None = None,
    ) -> SceneGraph:
        """从 3D 物体列表构建场景图.

        Args:
            object_list: M021 输出的 3D 物体列表.
            existing_graph: 已有图 (增量更新); None 则新建.

        Returns:
            SceneGraph.
        """
        graph = existing_graph or SceneGraph()
        graph.metadata["frame_id"] = object_list.frame_id
        graph.metadata["timestamp"] = object_list.timestamp
        graph.metadata["n_objects"] = object_list.n_objects

        # 1. 添加节点.
        node_ids: list[str] = []
        for obj in object_list.objects:
            node = self._object_to_node(obj)
            graph.add_node(node)
            node_ids.append(node.node_id)

        # 2. 计算关系 (两两之间).
        new_edges = self._compute_relations(object_list, graph)
        for edge in new_edges:
            graph.add_edge(edge)

        self._log.info(
            "scene graph built",
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            frame_id=object_list.frame_id,
        )
        return graph

    def _object_to_node(self, obj: Object3D) -> SceneNode:
        """Object3D → SceneNode."""
        return SceneNode(
            node_id=obj.object_id,
            label=obj.label,
            position=obj.center.copy(),
            size_3d=obj.size_3d.copy(),
            confidence=obj.confidence,
            metadata={
                "frame_id": obj.frame_id,
                "timestamp": obj.timestamp,
                "source_bbox_2d": obj.metadata.get("source_bbox_2d"),
                "depth_value": obj.metadata.get("depth_value"),
            },
        )

    def _compute_relations(self, object_list: Object3DList, graph: SceneGraph) -> list[SceneEdge]:
        """计算两两物体间的空间关系."""
        edges: list[SceneEdge] = []
        objects = object_list.objects
        n = len(objects)
        if n < _MIN_PAIRS:
            return edges

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                source = objects[i]
                target = objects[j]
                edge = self._compute_pair_relation(source, target)
                if edge is not None:
                    edges.append(edge)

        return edges

    def _compute_pair_relation(self, source: Object3D, target: Object3D) -> SceneEdge | None:
        """计算 source → target 的空间关系 (取最强关系)."""
        src_pos = source.center.astype(np.float64)
        tgt_pos = target.center.astype(np.float64)
        diff = tgt_pos - src_pos  # target 相对 source 的偏移.
        distance = float(np.linalg.norm(diff))

        if distance < _SAME_POS_EPS:
            return None  # 同位置, 跳过.

        relation: str | None = None
        metadata: dict[str, object] = {
            "distance": round(distance, 4),
        }

        # 1. 距离关系.
        if self.cfg.compute_distance_relations:
            if distance < self.cfg.next_to_threshold:
                relation = "next_to"
            elif distance < self.cfg.near_threshold:
                relation = "near"
            elif distance > self.cfg.far_threshold:
                relation = "far"

        # 2. 方向关系 (优先于距离).
        if self.cfg.compute_directions:
            direction_relation = self._compute_direction(diff)
            if direction_relation is not None:
                relation = direction_relation

        # 3. on 关系 (target 在 source 上方且距离近).
        if diff[_AXIS_Z] > self.cfg.on_threshold and distance < self.cfg.near_threshold:
            relation = "on"

        # 4. inside 关系 (target 在 source bbox 内).
        if self._is_inside(source, target):
            relation = "inside"

        if relation is None or relation not in RELATIONS:
            return None

        metadata["direction"] = {
            "dx": round(float(diff[_AXIS_X]), 4),
            "dy": round(float(diff[_AXIS_Y]), 4),
            "dz": round(float(diff[_AXIS_Z]), 4),
        }

        return SceneEdge(
            source_id=source.object_id,
            target_id=target.object_id,
            relation=relation,
            confidence=min(source.confidence, target.confidence),
            source="geometry",
            metadata=metadata,
        )

    def _compute_direction(self, diff: np.ndarray) -> str | None:
        """计算方向关系 (left/right/front/behind).

        Args:
            diff: target - source 的偏移向量 (3,).

        Returns:
            关系名 或 None.
        """
        if self.cfg.forward_axis == "x":
            forward_idx = _AXIS_X
            side_idx = _AXIS_Y
        else:
            forward_idx = _AXIS_Y
            side_idx = _AXIS_X

        forward = diff[forward_idx]
        side = diff[side_idx]

        # 判断主导方向 (forward vs side).
        if abs(forward) > abs(side):
            if forward > 0:
                return "in_front_of"
            return "behind"
        else:
            if side > 0:
                return "right_of" if self.cfg.forward_axis == "y" else "left_of"
            return "left_of" if self.cfg.forward_axis == "y" else "right_of"

    def _is_inside(self, source: Object3D, target: Object3D) -> bool:
        """判断 target 是否在 source 的 3D bbox 内."""
        src_center = source.center.astype(np.float64)
        tgt_center = target.center.astype(np.float64)
        src_half = source.size_3d.astype(np.float64) / 2.0
        diff = np.abs(tgt_center - src_center)
        return bool(np.all(diff <= src_half))
