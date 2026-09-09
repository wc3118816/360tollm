"""M036 集成测试: 完整 World Model 序列化端到端.

验证:
1. 完整流程: Image → Depth+Detection → Object3D → WorldModel+SceneGraph+SpatialMemory → JSON
2. AC: Schema 有版本号
3. AC: 向后兼容策略明确
4. 10 必需字段全覆盖
5. 序列化 → 反序列化 roundtrip
6. 文件持久化
"""

from __future__ import annotations

import json

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.objects import ObjectAssociator
from modules.scene_graph import SceneGraph, SceneGraphBuilder
from modules.serialization import (
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    WorldModelSerializer,
)
from modules.spatial_memory import SpatialMemory
from modules.vo.types import Pose
from modules.world_model import WorldModel


class TestFullPipeline:
    def _build_models(self) -> tuple[WorldModel, SceneGraph, SpatialMemory]:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        assoc = ObjectAssociator()

        model = WorldModel()
        sg_builder = SceneGraphBuilder()
        graph: SceneGraph | None = None
        mem = SpatialMemory()

        for i in range(3):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            model.add_objects(ol)
            graph = sg_builder.build(ol, existing_graph=graph)
            mem.observe(ol)

        if graph is None:
            graph = SceneGraph()
        return model, graph, mem

    def test_full_serialize(self) -> None:
        """完整序列化."""
        model, graph, mem = self._build_models()
        ser = WorldModelSerializer()
        json_str = ser.serialize(
            world_model=model,
            scene_graph=graph,
            spatial_memory=mem,
            pose=Pose.identity(),
            scene_id="test_scene",
            timestamp=5.0,
        )
        data = json.loads(json_str)
        assert data["scene_id"] == "test_scene"
        assert data["schema_version"] == SCHEMA_VERSION

    def test_ac_has_version(self) -> None:
        """AC: Schema 有版本号."""
        model, graph, mem = self._build_models()
        ser = WorldModelSerializer()
        json_str = ser.serialize(world_model=model, scene_graph=graph)
        data = json.loads(json_str)
        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_ac_backward_compatible(self) -> None:
        """AC: 向后兼容 — 旧 JSON 可解析."""
        # 模拟只含 scene_id + timestamp 的旧 JSON.
        old_json = json.dumps(
            {
                "scene_id": "legacy",
                "timestamp": 1.0,
            }
        )
        ser = WorldModelSerializer()
        data = ser.deserialize(old_json)
        # 所有缺失字段填默认值.
        for field in REQUIRED_FIELDS:
            assert field in data

    def test_all_10_fields_present(self) -> None:
        """10 必需字段全覆盖."""
        model, graph, mem = self._build_models()
        ser = WorldModelSerializer()
        json_str = ser.serialize(world_model=model, scene_graph=graph, spatial_memory=mem)
        data = json.loads(json_str)
        for field in REQUIRED_FIELDS:
            assert field in data, f"missing: {field}"

    def test_roundtrip(self) -> None:
        """序列化 → 反序列化 roundtrip."""
        model, graph, mem = self._build_models()
        ser = WorldModelSerializer()
        json_str = ser.serialize(
            world_model=model,
            scene_graph=graph,
            spatial_memory=mem,
            scene_id="roundtrip",
            timestamp=10.0,
        )
        data = ser.deserialize(json_str)
        assert data["scene_id"] == "roundtrip"
        assert data["timestamp"] == 10.0
        assert len(data["objects"]) > 0
        assert len(data["memory"]["objects"]) > 0

    def test_file_persistence(self, tmp_path) -> None:
        """文件持久化."""
        model, graph, mem = self._build_models()
        ser = WorldModelSerializer()
        file_path = str(tmp_path / "world_model.json")
        ser.serialize_to_file(
            file_path,
            world_model=model,
            scene_graph=graph,
            spatial_memory=mem,
            scene_id="file_persist",
        )
        data = ser.deserialize_from_file(file_path)
        assert data["scene_id"] == "file_persist"
        assert len(data["objects"]) > 0

    def test_multi_frame_serialize(self) -> None:
        """多帧序列化 (动态场景)."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        assoc = ObjectAssociator()
        model = WorldModel()
        sg_builder = SceneGraphBuilder()
        graph: SceneGraph | None = None
        mem = SpatialMemory()
        ser = WorldModelSerializer()

        for i in range(5):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            model.add_objects(ol)
            graph = sg_builder.build(ol, existing_graph=graph)
            mem.observe(ol)
            # 每帧序列化.
            json_str = ser.serialize(
                world_model=model,
                scene_graph=graph,
                spatial_memory=mem,
                scene_id="multi_frame",
                timestamp=float(i),
            )
            data = json.loads(json_str)
            assert data["schema_version"] == SCHEMA_VERSION
