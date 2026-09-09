"""M026 单元测试: SceneNode / SceneEdge / SceneGraph 数据结构.

覆盖:
- SceneNode 构造 + 校验 (position shape / confidence)
- SceneEdge 构造 + 校验 (relation 合法性 / confidence)
- SceneGraph add_node / add_edge / get_node / get_neighbors
- serialize / deserialize (AC: 图结构可序列化)
- summary / to_dict
- 边去重
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.scene_graph import SceneEdge, SceneGraph, SceneNode


def _make_node(
    node_id: str = "chair_000_00",
    label: str = "chair",
    position: np.ndarray | None = None,
    confidence: float = 0.9,
) -> SceneNode:
    if position is None:
        position = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    return SceneNode(node_id=node_id, label=label, position=position, confidence=confidence)


class TestSceneNode:
    def test_basic_construction(self) -> None:
        node = _make_node()
        assert node.node_id == "chair_000_00"
        assert node.label == "chair"
        assert node.confidence == 0.9

    def test_position_shape(self) -> None:
        with pytest.raises(ValueError, match="position"):
            SceneNode(node_id="x", label="x", position=np.array([1.0, 2.0]))

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_node(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            _make_node(confidence=-0.1)

    def test_to_dict(self) -> None:
        node = _make_node()
        d = node.to_dict()
        assert d["node_id"] == "chair_000_00"
        assert len(d["position"]) == 3


class TestSceneEdge:
    def test_basic_construction(self) -> None:
        edge = SceneEdge(source_id="a", target_id="b", relation="near", confidence=0.8)
        assert edge.relation == "near"
        assert edge.source == "geometry"  # 默认.

    def test_invalid_relation(self) -> None:
        with pytest.raises(ValueError, match="relation"):
            SceneEdge(source_id="a", target_id="b", relation="unknown_relation")

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SceneEdge(source_id="a", target_id="b", relation="near", confidence=1.5)

    def test_to_dict(self) -> None:
        edge = SceneEdge(source_id="a", target_id="b", relation="on")
        d = edge.to_dict()
        assert d["source_id"] == "a"
        assert d["relation"] == "on"


class TestSceneGraph:
    def test_empty_graph(self) -> None:
        graph = SceneGraph()
        assert graph.n_nodes == 0
        assert graph.n_edges == 0

    def test_add_node(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("chair_000_00"))
        assert graph.n_nodes == 1
        assert graph.get_node("chair_000_00") is not None

    def test_add_edge(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("a"))
        graph.add_node(_make_node("b"))
        edge = SceneEdge(source_id="a", target_id="b", relation="near")
        graph.add_edge(edge)
        assert graph.n_edges == 1

    def test_edge_dedup(self) -> None:
        """同 source-target-relation 的边去重."""
        graph = SceneGraph()
        edge1 = SceneEdge(source_id="a", target_id="b", relation="near")
        edge2 = SceneEdge(source_id="a", target_id="b", relation="near")
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        assert graph.n_edges == 1  # 去重.

    def test_get_neighbors(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("a"))
        graph.add_node(_make_node("b"))
        graph.add_node(_make_node("c"))
        graph.add_edge(SceneEdge(source_id="a", target_id="b", relation="near"))
        graph.add_edge(SceneEdge(source_id="a", target_id="c", relation="far"))
        neighbors = graph.get_neighbors("a")
        assert len(neighbors) == 2

    def test_get_edges_from(self) -> None:
        graph = SceneGraph()
        graph.add_edge(SceneEdge(source_id="a", target_id="b", relation="near"))
        graph.add_edge(SceneEdge(source_id="a", target_id="c", relation="far"))
        edges = graph.get_edges_from("a")
        assert len(edges) == 2

    def test_filter_by_label(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("a", label="chair"))
        graph.add_node(_make_node("b", label="table"))
        chairs = graph.filter_by_label("chair")
        assert len(chairs) == 1

    def test_summary(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("a", label="chair"))
        graph.add_node(_make_node("b", label="table"))
        graph.add_edge(SceneEdge(source_id="a", target_id="b", relation="near"))
        s = graph.summary()
        assert s["n_nodes"] == 2
        assert s["n_edges"] == 1
        assert "near" in s["relations"]


class TestSerialize:
    """AC: 图结构可序列化."""

    def test_serialize_deserialize_roundtrip(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("a", label="chair"))
        graph.add_node(_make_node("b", label="table"))
        graph.add_edge(SceneEdge(source_id="a", target_id="b", relation="near", confidence=0.8))
        graph.metadata["frame_id"] = 5

        # 序列化.
        json_str = graph.serialize()
        assert isinstance(json_str, str)

        # 反序列化.
        graph2 = SceneGraph.deserialize(json_str)
        assert graph2.n_nodes == 2
        assert graph2.n_edges == 1
        assert graph2.metadata["frame_id"] == 5

        # 节点保持.
        node_a = graph2.get_node("a")
        assert node_a is not None
        assert node_a.label == "chair"

        # 边保持.
        edge = graph2.edges[0]
        assert edge.relation == "near"
        assert edge.confidence == 0.8

    def test_serialize_valid_json(self) -> None:
        """序列化输出是合法 JSON."""
        graph = SceneGraph()
        graph.add_node(_make_node("a"))
        json_str = graph.serialize()
        data = json.loads(json_str)  # 不抛异常 = 合法.
        assert "nodes" in data
        assert "edges" in data

    def test_to_dict(self) -> None:
        graph = SceneGraph()
        graph.add_node(_make_node("a"))
        d = graph.to_dict()
        assert len(d["nodes"]) == 1
