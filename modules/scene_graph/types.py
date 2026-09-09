"""M026 Scene Graph: 节点和边组成的场景图.

SceneNode:
- node_id: str 唯一标识.
- label: str 类别名 (chair / table / person / room / door).
- position: float32 (3,) 世界坐标 [x, y, z] (米).
- size_3d: float32 (3,) 尺寸 [dx, dy, dz] (米, 可选).
- confidence: float ∈ [0, 1].
- metadata: 扩展字段 (source_object_id / frame_id / timestamp).

SceneEdge:
- source_id: str 起点节点 ID.
- target_id: str 终点节点 ID.
- relation: str 关系名 (inside / on / next_to / near / far / left_of / right_of / ...).
- confidence: float ∈ [0, 1].
- source: str 关系来源 (geometry / detection / manual).
- metadata: 扩展字段 (distance / direction 等).

SceneGraph:
- nodes: dict[str, SceneNode].
- edges: list[SceneEdge].
- add_node / add_edge / get_node / get_neighbors / serialize / deserialize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_NDIM_3D = 3
_SIZE_DIM = 3
_DEFAULT_CONF = 1.0

# 支持的关系类型.
RELATIONS = {
    "inside",
    "on",
    "next_to",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "near",
    "far",
    "connected_to",
}


@dataclass(slots=True)
class SceneNode:
    """场景图节点 (代表一个物体/区域).

    Attributes:
        node_id: 唯一标识.
        label: 类别名.
        position: float32 (3,) 世界坐标.
        size_3d: float32 (3,) 尺寸 (可选).
        confidence: 置信度 ∈ [0, 1].
        metadata: 扩展字段.
    """

    node_id: str
    label: str
    position: np.ndarray
    size_3d: np.ndarray = field(default_factory=lambda: np.zeros(_SIZE_DIM, dtype=np.float32))
    confidence: float = _DEFAULT_CONF
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).ravel()
        if self.position.shape[0] != _NDIM_3D:
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        self.size_3d = np.asarray(self.size_3d, dtype=np.float32).ravel()
        if self.size_3d.shape[0] != _SIZE_DIM:
            raise ValueError(f"size_3d must be (3,), got {self.size_3d.shape}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be ∈ [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict (JSON 兼容)."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "position": self.position.tolist(),
            "size_3d": self.size_3d.tolist(),
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SceneEdge:
    """场景图边 (代表两个节点间的关系).

    Attributes:
        source_id: 起点节点 ID.
        target_id: 终点节点 ID.
        relation: 关系名.
        confidence: 置信度 ∈ [0, 1].
        source: 关系来源 (geometry / detection / manual).
        metadata: 扩展字段 (distance / direction 等).
    """

    source_id: str
    target_id: str
    relation: str
    confidence: float = _DEFAULT_CONF
    source: str = "geometry"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}, got '{self.relation}'")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be ∈ [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict (JSON 兼容)."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class SceneGraph:
    """场景图 (节点 + 边).

    Attributes:
        nodes: dict[str, SceneNode] node_id → node.
        edges: list[SceneEdge].
        metadata: 图级元数据 (frame_id / timestamp / builder 等).
    """

    nodes: dict[str, SceneNode] = field(default_factory=dict)
    edges: list[SceneEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def labels(self) -> list[str]:
        return list({n.label for n in self.nodes.values()})

    def add_node(self, node: SceneNode) -> None:
        """添加节点 (重复 ID 覆盖)."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: SceneEdge) -> None:
        """添加边 (自动去重同 source-target-relation)."""
        for existing in self.edges:
            if (
                existing.source_id == edge.source_id
                and existing.target_id == edge.target_id
                and existing.relation == edge.relation
            ):
                return  # 已存在, 跳过.
        self.edges.append(edge)

    def get_node(self, node_id: str) -> SceneNode | None:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> list[SceneNode]:
        """获取与指定节点有边相连的邻居节点."""
        neighbor_ids = {
            edge.target_id
            for edge in self.edges
            if edge.source_id == node_id and edge.target_id in self.nodes
        }
        return [self.nodes[nid] for nid in neighbor_ids]

    def get_edges_from(self, node_id: str) -> list[SceneEdge]:
        """获取从指定节点出发的边."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_between(self, source_id: str, target_id: str) -> list[SceneEdge]:
        """获取两节点间的所有边."""
        return [e for e in self.edges if e.source_id == source_id and e.target_id == target_id]

    def filter_by_label(self, label: str) -> list[SceneNode]:
        """按类别过滤节点."""
        return [n for n in self.nodes.values() if n.label == label]

    def summary(self) -> dict[str, Any]:
        """统计摘要."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "labels": self.labels,
            "relations": list({e.relation for e in self.edges}),
            "metadata": self.metadata,
        }

    def serialize(self) -> str:
        """序列化为 JSON 字符串 (AC: 图结构可序列化)."""
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, json_str: str) -> SceneGraph:
        """从 JSON 字符串反序列化."""
        data = json.loads(json_str)
        graph = cls(metadata=data.get("metadata", {}))
        for n_dict in data.get("nodes", []):
            node = SceneNode(
                node_id=n_dict["node_id"],
                label=n_dict["label"],
                position=np.array(n_dict["position"], dtype=np.float32),
                size_3d=np.array(n_dict.get("size_3d", [0, 0, 0]), dtype=np.float32),
                confidence=n_dict.get("confidence", 1.0),
                metadata=n_dict.get("metadata", {}),
            )
            graph.add_node(node)
        for e_dict in data.get("edges", []):
            edge = SceneEdge(
                source_id=e_dict["source_id"],
                target_id=e_dict["target_id"],
                relation=e_dict["relation"],
                confidence=e_dict.get("confidence", 1.0),
                source=e_dict.get("source", "geometry"),
                metadata=e_dict.get("metadata", {}),
            )
            graph.edges.append(edge)
        return graph
