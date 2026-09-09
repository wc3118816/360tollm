"""M036 单元测试: Schema + Serializer.

覆盖:
- SCHEMA_VERSION 存在 + 格式 (SemVer)
- REQUIRED_FIELDS 包含 10 个字段
- SchemaVersion.is_compatible 版本兼容检查
- validate_schema 验证
- create_empty_schema 默认值
- WorldModelSerializer.serialize 输出合法 JSON
- WorldModelSerializer.deserialize 向后兼容 (AC)
- 10 必需字段全覆盖
- 空输入序列化
- 文件 I/O
"""

from __future__ import annotations

import json

import numpy as np

from modules.serialization import (
    COORDINATE_SYSTEM,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    SchemaVersion,
    WorldModelSerializer,
    create_empty_schema,
    validate_schema,
)


class TestSchema:
    def test_schema_version_format(self) -> None:
        """AC: Schema 有版本号 (SemVer 格式)."""
        assert SCHEMA_VERSION == "1.0.0"
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()

    def test_required_fields_count(self) -> None:
        """必需字段 = 10 个."""
        assert len(REQUIRED_FIELDS) == 10

    def test_required_fields_content(self) -> None:
        for field in (
            "scene_id",
            "coordinate_system",
            "camera_pose",
            "objects",
            "relations",
            "regions",
            "occupancy",
            "memory",
            "confidence",
            "timestamp",
        ):
            assert field in REQUIRED_FIELDS

    def test_coordinate_system(self) -> None:
        assert COORDINATE_SYSTEM == "world_xyz_meters"


class TestSchemaVersion:
    def test_current_version(self) -> None:
        assert SchemaVersion.current() == SchemaVersion.V1_0_0
        assert SchemaVersion.current().value == SCHEMA_VERSION

    def test_is_compatible_same(self) -> None:
        assert SchemaVersion.is_compatible("1.0.0")

    def test_is_compatible_minor(self) -> None:
        """同 major → 兼容."""
        assert SchemaVersion.is_compatible("1.5.3")

    def test_is_compatible_major_diff(self) -> None:
        """不同 major → 不兼容."""
        assert not SchemaVersion.is_compatible("2.0.0")

    def test_is_compatible_invalid(self) -> None:
        assert not SchemaVersion.is_compatible("invalid")


class TestValidateSchema:
    def test_valid_empty(self) -> None:
        data = create_empty_schema()
        errors = validate_schema(data)
        assert errors == []

    def test_missing_field(self) -> None:
        data = create_empty_schema()
        del data["scene_id"]
        errors = validate_schema(data)
        assert any("scene_id" in e for e in errors)

    def test_incompatible_version(self) -> None:
        data = create_empty_schema()
        data["schema_version"] = "2.0.0"
        errors = validate_schema(data)
        assert any("incompatible" in e for e in errors)


class TestCreateEmptySchema:
    def test_has_all_fields(self) -> None:
        data = create_empty_schema()
        for field in REQUIRED_FIELDS:
            assert field in data
        assert "schema_version" in data

    def test_defaults(self) -> None:
        data = create_empty_schema("test_scene")
        assert data["scene_id"] == "test_scene"
        assert data["objects"] == []
        assert data["relations"] == []
        assert data["regions"] == []
        assert data["confidence"] == 0.0


class TestSerializer:
    def test_serialize_empty(self) -> None:
        """空输入 → 合法 JSON."""
        ser = WorldModelSerializer()
        json_str = ser.serialize(scene_id="empty", timestamp=1.0)
        data = json.loads(json_str)
        for field in REQUIRED_FIELDS:
            assert field in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_serialize_returns_json_string(self) -> None:
        ser = WorldModelSerializer()
        json_str = ser.serialize()
        assert isinstance(json_str, str)
        # 合法 JSON.
        json.loads(json_str)

    def test_serialize_all_10_fields(self) -> None:
        """AC: 10 个必需字段全覆盖."""
        ser = WorldModelSerializer()
        json_str = ser.serialize(scene_id="test", timestamp=1.0)
        data = json.loads(json_str)
        for field in REQUIRED_FIELDS:
            assert field in data, f"missing field: {field}"

    def test_serialize_with_version(self) -> None:
        """AC: Schema 有版本号."""
        ser = WorldModelSerializer()
        json_str = ser.serialize()
        data = json.loads(json_str)
        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION

    def test_deserialize_roundtrip(self) -> None:
        """序列化 → 反序列化 roundtrip."""
        ser = WorldModelSerializer()
        json_str = ser.serialize(scene_id="rt", timestamp=5.0)
        data = ser.deserialize(json_str)
        assert data["scene_id"] == "rt"
        assert data["timestamp"] == 5.0
        assert data["schema_version"] == SCHEMA_VERSION

    def test_deserialize_backward_compatible(self) -> None:
        """AC: 向后兼容 — 缺失字段填默认值."""
        # 模拟旧版本 JSON (缺失部分字段).
        old_json = json.dumps(
            {
                "schema_version": "1.0.0",
                "scene_id": "old_scene",
                "timestamp": 1.0,
                # 缺失: coordinate_system, camera_pose, objects, relations, regions, occupancy, memory, confidence
            }
        )
        ser = WorldModelSerializer()
        data = ser.deserialize(old_json)
        # 缺失字段应填默认值.
        assert data["scene_id"] == "old_scene"
        assert data["objects"] == []
        assert data["relations"] == []
        assert data["confidence"] == 0.0
        assert data["coordinate_system"] == COORDINATE_SYSTEM

    def test_deserialize_unknown_version(self) -> None:
        """未知版本 → warning 但不报错."""
        json_str = json.dumps(
            {
                "schema_version": "1.99.0",  # 同 major, 兼容.
                "scene_id": "future",
            }
        )
        ser = WorldModelSerializer()
        data = ser.deserialize(json_str)
        assert data["scene_id"] == "future"

    def test_deserialize_incompatible_version(self) -> None:
        """不兼容版本 → warning 但仍解析."""
        json_str = json.dumps(
            {
                "schema_version": "2.0.0",  # 不同 major.
                "scene_id": "v2_scene",
            }
        )
        ser = WorldModelSerializer()
        data = ser.deserialize(json_str)
        # 仍能解析 (向后兼容策略).
        assert data["scene_id"] == "v2_scene"
        # 但版本记录保留.
        assert data["schema_version"] == "2.0.0"

    def test_serialize_with_pose(self) -> None:
        from modules.vo.types import Pose

        ser = WorldModelSerializer()
        pose = Pose.identity()
        json_str = ser.serialize(pose=pose, scene_id="test", timestamp=1.0)
        data = json.loads(json_str)
        assert data["camera_pose"] is not None
        assert "R" in data["camera_pose"]
        assert "t" in data["camera_pose"]

    def test_serialize_with_world_model(self) -> None:
        from modules.objects import Object3D, Object3DList
        from modules.world_model import WorldModel

        ol = Object3DList(
            objects=[
                Object3D(
                    object_id="chair_001",
                    label="chair",
                    center=np.array([1, 2, 3], dtype=np.float32),
                    confidence=0.9,
                )
            ]
        )
        model = WorldModel()
        model.add_objects(ol)
        ser = WorldModelSerializer()
        json_str = ser.serialize(world_model=model, scene_id="test")
        data = json.loads(json_str)
        assert len(data["objects"]) == 1
        assert data["objects"][0]["object_id"] == "chair_001"
        assert data["confidence"] == 0.9

    def test_serialize_with_scene_graph(self) -> None:
        from modules.scene_graph import SceneEdge, SceneGraph, SceneNode

        graph = SceneGraph()
        graph.add_node(
            SceneNode(node_id="a", label="chair", position=np.array([0, 0, 0], dtype=np.float32))
        )
        graph.add_node(
            SceneNode(node_id="b", label="table", position=np.array([1, 0, 0], dtype=np.float32))
        )
        graph.add_edge(SceneEdge(source_id="a", target_id="b", relation="near"))
        ser = WorldModelSerializer()
        json_str = ser.serialize(scene_graph=graph, scene_id="test")
        data = json.loads(json_str)
        assert len(data["relations"]) == 1
        assert data["relations"][0]["relation"] == "near"

    def test_serialize_with_spatial_memory(self) -> None:
        from modules.objects import Object3D, Object3DList
        from modules.spatial_memory import SpatialMemory

        ol = Object3DList(
            objects=[
                Object3D(
                    object_id="obj_000_00",
                    label="chair",
                    center=np.array([1, 0, 0], dtype=np.float32),
                    confidence=0.9,
                )
            ],
            frame_id=0,
            timestamp=1.0,
        )
        mem = SpatialMemory()
        mem.observe(ol)
        ser = WorldModelSerializer()
        json_str = ser.serialize(spatial_memory=mem, scene_id="test")
        data = json.loads(json_str)
        assert "memory" in data
        assert data["memory"]["n_objects"] == 1

    def test_file_io(self, tmp_path) -> None:
        """文件 I/O."""
        ser = WorldModelSerializer()
        file_path = str(tmp_path / "world_model.json")
        ser.serialize_to_file(file_path, scene_id="file_test", timestamp=1.0)
        data = ser.deserialize_from_file(file_path)
        assert data["scene_id"] == "file_test"
