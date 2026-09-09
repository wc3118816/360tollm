"""M036 World Model 序列化: 统一序列化器.

WorldModelSerializer:
- serialize(world_model, scene_graph, spatial_memory, pose, scene_id) → JSON 字符串.
- deserialize(json_str) → dict (含所有字段).
- 支持 SceneGraph + SpatialMemory + WorldModel 统一序列化.

输出 JSON Schema (10 必需字段 + schema_version):
{
  "schema_version": "1.0.0",
  "scene_id": "room_001",
  "coordinate_system": "world_xyz_meters",
  "camera_pose": {...},
  "objects": [...],
  "relations": [...],
  "regions": [],
  "occupancy": {...},
  "memory": {"objects": [...]},
  "confidence": 0.9,
  "timestamp": 123.45
}

用法:
    ser = WorldModelSerializer()
    json_str = ser.serialize(world_model, scene_graph, spatial_memory, pose, "room_001")
    data = ser.deserialize(json_str)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.logging import get_logger
from modules.scene_graph import SceneGraph
from modules.spatial_memory import SpatialMemory
from modules.vo.types import Pose
from modules.world_model import WorldModel

from .schema import (
    COORDINATE_SYSTEM,
    MIN_COMPATIBLE_VERSION,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    SchemaVersion,
    create_empty_schema,
    validate_schema,
)

_LOG = get_logger("modules.serialization")

_NDIM_3D = 3
_SE3_SIZE = 4


@dataclass
class WorldModelSerializer:
    """World Model 统一序列化器.

    序列化 WorldModel + SceneGraph + SpatialMemory 为统一 JSON.
    """

    def __post_init__(self) -> None:
        self._log = _LOG

    def serialize(  # noqa: PLR0917
        self,
        world_model: WorldModel | None = None,
        scene_graph: SceneGraph | None = None,
        spatial_memory: SpatialMemory | None = None,
        pose: Pose | None = None,
        scene_id: str = "default",
        timestamp: float | None = None,
    ) -> str:
        """序列化为 JSON 字符串.

        Args:
            world_model: WorldModel (M029).
            scene_graph: SceneGraph (M026).
            spatial_memory: SpatialMemory (M031).
            pose: 当前相机位姿.
            scene_id: 场景 ID.
            timestamp: 时间戳 (None 用 time.time()).

        Returns:
            str JSON 字符串.
        """
        if timestamp is None:
            timestamp = time.time()

        data = create_empty_schema(scene_id)
        data["schema_version"] = SCHEMA_VERSION
        data["timestamp"] = float(timestamp)
        data["coordinate_system"] = COORDINATE_SYSTEM

        # camera_pose.
        if pose is not None:
            data["camera_pose"] = self._serialize_pose(pose)
        else:
            data["camera_pose"] = None

        # objects (从 WorldModel).
        if world_model is not None:
            data["objects"] = [self._serialize_object(obj) for obj in world_model.objects.values()]
            data["confidence"] = (
                float(np.mean([o.confidence for o in world_model.objects.values()]))
                if world_model.n_objects > 0
                else 0.0
            )

        # relations (从 SceneGraph).
        if scene_graph is not None:
            data["relations"] = [e.to_dict() for e in scene_graph.edges]
            # 如果 SceneGraph 有节点但 WorldModel 为空, 也序列化节点.
            if world_model is None and scene_graph.n_nodes > 0:
                data["objects"] = [n.to_dict() for n in scene_graph.nodes.values()]

        # memory (从 SpatialMemory).
        if spatial_memory is not None:
            data["memory"] = {
                "objects": [t.to_dict() for t in spatial_memory.objects.values()],
                "config": {
                    "association_threshold": spatial_memory.cfg.association_threshold,
                    "max_age": spatial_memory.cfg.max_age,
                },
                "n_objects": spatial_memory.n_objects,
            }

        # occupancy (元数据, 实际占用栅格由 M018 处理).
        data["occupancy"] = {
            "supported": True,
            "module": "modules.occupancy",
            "note": "occupancy grid serialized separately",
        }

        # regions (当前为空, M025 语义地图未实现).
        data["regions"] = []

        # 验证.
        errors = validate_schema(data)
        if errors:
            self._log.warning("schema validation errors", errors=errors)

        return json.dumps(data, ensure_ascii=False, indent=2)

    def deserialize(self, json_str: str) -> dict[str, Any]:
        """从 JSON 字符串反序列化.

        向后兼容策略:
        - 缺失字段填默认值.
        - 版本不匹配 → warning 但不报错.
        - 多余字段保留 (forward compatible).

        Args:
            json_str: JSON 字符串.

        Returns:
            dict 完整 Schema 数据.
        """
        data = json.loads(json_str)

        # 版本检查.
        version = str(data.get("schema_version", "unknown"))
        if version == "unknown":
            self._log.warning("missing schema_version, assuming compatible")
        elif not SchemaVersion.is_compatible(version):
            self._log.warning(
                "schema version may be incompatible",
                got=version,
                current=SCHEMA_VERSION,
                min_compatible=MIN_COMPATIBLE_VERSION,
            )

        # 向后兼容: 缺失字段填默认值.
        empty = create_empty_schema()
        for field_name in REQUIRED_FIELDS:
            if field_name not in data:
                data[field_name] = empty[field_name]
                self._log.debug("filled missing field", field=field_name)

        # 确保 schema_version 存在.
        if "schema_version" not in data:
            data["schema_version"] = SCHEMA_VERSION

        return data

    def serialize_to_file(  # noqa: PLR0917
        self,
        file_path: str,
        world_model: WorldModel | None = None,
        scene_graph: SceneGraph | None = None,
        spatial_memory: SpatialMemory | None = None,
        pose: Pose | None = None,
        scene_id: str = "default",
        timestamp: float | None = None,
    ) -> None:
        """序列化到文件."""
        json_str = self.serialize(
            world_model=world_model,
            scene_graph=scene_graph,
            spatial_memory=spatial_memory,
            pose=pose,
            scene_id=scene_id,
            timestamp=timestamp,
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        self._log.info("serialized to file", path=file_path, size=len(json_str))

    def deserialize_from_file(self, file_path: str) -> dict[str, Any]:
        """从文件反序列化."""
        with open(file_path, encoding="utf-8") as f:
            json_str = f.read()
        return self.deserialize(json_str)

    def _serialize_pose(self, pose: Pose) -> dict[str, Any]:
        """序列化 Pose."""
        return {
            "frame_id": int(pose.frame_id),
            "R": pose.R.tolist(),
            "t": pose.t.tolist(),
            "matrix": pose.matrix.tolist(),
        }

    def _serialize_object(self, obj: Any) -> dict[str, Any]:
        """序列化 Object3D → dict."""
        return obj.to_dict()
