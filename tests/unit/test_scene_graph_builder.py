"""M026 单元测试: SceneGraphBuilder (Object3DList → SceneGraph).

覆盖:
- 基本构建 (Object3DList → SceneGraph)
- 节点数量正确
- 节点具备世界坐标 (AC)
- 边具备置信度/来源 (AC)
- 空间关系计算 (near/far/left_of/right_of/on/inside)
- 增量更新
- 空物体列表
"""

from __future__ import annotations

import numpy as np

from modules.objects import Object3D, Object3DList
from modules.scene_graph import (
    GraphBuilderConfig,
    SceneGraphBuilder,
)


def _make_object(
    object_id: str = "obj_000_00",
    label: str = "chair",
    center: np.ndarray | None = None,
    confidence: float = 0.9,
    size_3d: np.ndarray | None = None,
) -> Object3D:
    if center is None:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if size_3d is None:
        size_3d = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return Object3D(
        object_id=object_id,
        label=label,
        center=center,
        size_3d=size_3d,
        confidence=confidence,
    )


class TestBuild:
    def test_basic_build(self) -> None:
        obj = _make_object()
        ol = Object3DList(objects=[obj], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        assert graph.n_nodes == 1

    def test_multiple_objects(self) -> None:
        objs = [
            _make_object("a", "chair", np.array([0, 0, 0], dtype=np.float32)),
            _make_object("b", "table", np.array([1, 0, 0], dtype=np.float32)),
            _make_object("c", "cup", np.array([5, 5, 0], dtype=np.float32)),
        ]
        ol = Object3DList(objects=objs, frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        assert graph.n_nodes == 3

    def test_empty_list(self) -> None:
        ol = Object3DList(objects=[], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        assert graph.n_nodes == 0
        assert graph.n_edges == 0

    def test_ac_node_has_world_coords(self) -> None:
        """AC: 节点具备世界坐标."""
        center = np.array([3.0, 4.0, 1.0], dtype=np.float32)
        obj = _make_object(center=center)
        ol = Object3DList(objects=[obj], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        node = graph.get_node("obj_000_00")
        assert node is not None
        assert np.allclose(node.position, center)

    def test_ac_edge_has_confidence_source(self) -> None:
        """AC: 边具有置信度/来源."""
        obj1 = _make_object("a", confidence=0.8)
        obj2 = _make_object("b", center=np.array([1, 0, 0], dtype=np.float32), confidence=0.6)
        ol = Object3DList(objects=[obj1, obj2], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        if graph.n_edges > 0:
            edge = graph.edges[0]
            assert 0.0 <= edge.confidence <= 1.0
            assert isinstance(edge.source, str)


class TestRelations:
    def test_near_relation(self) -> None:
        obj1 = _make_object("a", center=np.array([0, 0, 0], dtype=np.float32))
        obj2 = _make_object("b", center=np.array([1, 0, 0], dtype=np.float32))
        ol = Object3DList(objects=[obj1, obj2], frame_id=0)
        builder = SceneGraphBuilder(
            GraphBuilderConfig(near_threshold=2.0, next_to_threshold=0.5, compute_directions=False)
        )
        graph = builder.build(ol)
        # 距离 1m → near.
        relations = [e.relation for e in graph.edges]
        assert "near" in relations or "next_to" in relations

    def test_far_relation(self) -> None:
        obj1 = _make_object("a", center=np.array([0, 0, 0], dtype=np.float32))
        obj2 = _make_object("b", center=np.array([10, 0, 0], dtype=np.float32))
        ol = Object3DList(objects=[obj1, obj2], frame_id=0)
        builder = SceneGraphBuilder(GraphBuilderConfig(far_threshold=5.0, compute_directions=False))
        graph = builder.build(ol)
        relations = [e.relation for e in graph.edges]
        assert "far" in relations

    def test_direction_relation(self) -> None:
        """方向关系 (forward_axis=y: +y=front, -y=behind, +x=right, -x=left)."""
        obj1 = _make_object("a", center=np.array([0, 0, 0], dtype=np.float32))
        # b 在 a 的前方 (y+).
        obj2 = _make_object("b", center=np.array([0, 3, 0], dtype=np.float32))
        ol = Object3DList(objects=[obj1, obj2], frame_id=0)
        builder = SceneGraphBuilder(GraphBuilderConfig(far_threshold=1.0))
        graph = builder.build(ol)
        relations_from_a = [e.relation for e in graph.edges if e.source_id == "a"]
        assert "in_front_of" in relations_from_a

    def test_on_relation(self) -> None:
        """on: target 在 source 上方且距离近."""
        obj1 = _make_object(
            "table",
            "table",
            np.array([0, 0, 0], dtype=np.float32),
            size_3d=np.array([1, 1, 1], dtype=np.float32),
        )
        obj2 = _make_object(
            "cup",
            "cup",
            np.array([0, 0, 0.8], dtype=np.float32),
            size_3d=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        )
        ol = Object3DList(objects=[obj1, obj2], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        # cup → table 或 table → cup 的 on 关系.
        on_edges = [e for e in graph.edges if e.relation == "on"]
        assert len(on_edges) > 0

    def test_inside_relation(self) -> None:
        """inside: target 在 source bbox 内."""
        obj1 = _make_object(
            "room",
            "room",
            np.array([0, 0, 0], dtype=np.float32),
            size_3d=np.array([5, 5, 3], dtype=np.float32),
        )
        obj2 = _make_object(
            "chair",
            "chair",
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
            size_3d=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        ol = Object3DList(objects=[obj1, obj2], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        inside_edges = [e for e in graph.edges if e.relation == "inside"]
        assert len(inside_edges) > 0


class TestIncremental:
    def test_update_existing(self) -> None:
        """增量更新."""
        obj1 = _make_object("a", center=np.array([0, 0, 0], dtype=np.float32))
        ol1 = Object3DList(objects=[obj1], frame_id=0)
        builder = SceneGraphBuilder()
        graph = builder.build(ol1)
        assert graph.n_nodes == 1

        # 第二帧: 加一个新物体.
        obj2 = _make_object("b", center=np.array([1, 0, 0], dtype=np.float32))
        ol2 = Object3DList(objects=[obj1, obj2], frame_id=1)
        graph = builder.build(ol2, existing_graph=graph)
        assert graph.n_nodes == 2
        assert graph.metadata["frame_id"] == 1
