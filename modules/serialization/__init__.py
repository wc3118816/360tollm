"""M036 World Model 序列化子包.

公开 API:
    WorldModelSerializer - 统一序列化器 (WorldModel+SceneGraph+SpatialMemory → JSON)
    SchemaVersion         - Schema 版本枚举
    SCHEMA_VERSION        - 当前版本号 "1.0.0"
    REQUIRED_FIELDS       - 必需字段元组
    COORDINATE_SYSTEM     - 坐标系约定
    validate_schema       - Schema 验证函数
    create_empty_schema   - 创建空 Schema (向后兼容)

用法:
    from modules.serialization import WorldModelSerializer
    ser = WorldModelSerializer()
    json_str = ser.serialize(world_model, scene_graph, spatial_memory, pose, "room_001")
    data = ser.deserialize(json_str)  # AC: 向后兼容
"""

from __future__ import annotations

from .schema import (
    COORDINATE_SYSTEM,
    MIN_COMPATIBLE_VERSION,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    SchemaVersion,
    create_empty_schema,
    validate_schema,
)
from .serializer import WorldModelSerializer

__all__ = [
    "WorldModelSerializer",
    "SchemaVersion",
    "SCHEMA_VERSION",
    "MIN_COMPATIBLE_VERSION",
    "REQUIRED_FIELDS",
    "COORDINATE_SYSTEM",
    "validate_schema",
    "create_empty_schema",
]
