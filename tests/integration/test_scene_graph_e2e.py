"""M026 集成测试: Image → Depth+Detection → 3D Object → Scene Graph 端到端.

验证:
1. 完整流程: Image → Depth + Detection → Object3D → SceneGraph
2. AC: 图结构可序列化
3. AC: 节点具备世界坐标
4. AC: 边具有置信度/来源
5. SceneGraph JSON → 反序列化恢复
"""

from __future__ import annotations

import json

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.objects import ObjectAssociator
from modules.scene_graph import SceneGraph, SceneGraphBuilder
from modules.vo.types import Pose


class TestFullPipeline:
    def test_image_to_scene_graph(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        # 深度.
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        # 检测.
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        dl = det.detect(img)
        # 关联.
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        # 构建场景图.
        builder = SceneGraphBuilder()
        graph = builder.build(ol)
        assert graph.n_nodes == ol.n_objects
        assert graph.n_nodes > 0

    def test_ac_serializable(self) -> None:
        """AC: 图结构可序列化."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        builder = SceneGraphBuilder()
        graph = builder.build(ol)

        # 序列化.
        json_str = graph.serialize()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "nodes" in data
        assert "edges" in data

    def test_ac_node_world_coords(self) -> None:
        """AC: 节点具备世界坐标."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        builder = SceneGraphBuilder()
        graph = builder.build(ol)

        for node in graph.nodes.values():
            assert node.position.shape == (3,)
            # 世界坐标应在合理范围.
            dist = float(np.linalg.norm(node.position))
            assert 0.1 < dist < 100.0

    def test_ac_edge_confidence_source(self) -> None:
        """AC: 边具有置信度/来源."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=4, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        builder = SceneGraphBuilder()
        graph = builder.build(ol)

        for edge in graph.edges:
            assert 0.0 <= edge.confidence <= 1.0
            assert isinstance(edge.source, str)
            assert edge.source == "geometry"  # 当前来源.

    def test_roundtrip(self) -> None:
        """序列化 → 反序列化 → 恢复."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        builder = SceneGraphBuilder()
        graph = builder.build(ol)

        # 序列化 → 反序列化.
        json_str = graph.serialize()
        graph2 = SceneGraph.deserialize(json_str)
        assert graph2.n_nodes == graph.n_nodes
        assert graph2.n_edges == graph.n_edges

    def test_multi_frame_build(self) -> None:
        """多帧增量构建."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        assoc = ObjectAssociator()
        builder = SceneGraphBuilder()
        graph = None

        for i in range(3):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            graph = builder.build(ol, existing_graph=graph)

        assert graph is not None
        assert graph.n_nodes > 0
        # 序列化仍有效.
        assert len(graph.serialize()) > 0
