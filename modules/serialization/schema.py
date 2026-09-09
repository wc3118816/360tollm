"""M036 World Model 序列化: JSON Schema 定义 + 版本管理.

Schema 字段 (LLM 可消费):
1. scene_id: str 场景唯一标识.
2. coordinate_system: str 坐标系约定 (如 "world_xyz_meters").
3. camera_pose: dict 相机位姿 (R + t + frame_id).
4. objects: list[dict] 3D 物体列表.
5. relations: list[dict] 物体间关系 (SceneGraph 边).
6. regions: list[dict] 空间区域 (可选, 当前为空).
7. occupancy: dict 占用栅格 (可选, 当前为元数据).
8. memory: dict 空间记忆 (TrackedObject 列表).
9. confidence: float 全局置信度.
10. timestamp: float 序列化时间戳.

版本管理 (AC: 向后兼容策略明确):
- SCHEMA_VERSION = "1.0.0" (SemVer).
- 向后兼容: 新版本可添加字段, 但不删除/重命名旧字段.
- 旧版本反序列化时: 缺失字段填默认值, 不报错.
- 版本检查: deserialize 时记录 schema_version, 与当前不匹配则 warning.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# Schema 版本 (SemVer: major.minor.patch).
SCHEMA_VERSION = "1.0.0"

# 支持的最小版本 (向后兼容到此版本).
MIN_COMPATIBLE_VERSION = "1.0.0"

# Schema 必须包含的 10 个字段.
REQUIRED_FIELDS = (
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
)

# 坐标系约定.
COORDINATE_SYSTEM = "world_xyz_meters"


class SchemaVersion(str, Enum):  # noqa: UP042
    """Schema 版本枚举."""

    V1_0_0 = "1.0.0"

    @classmethod
    def current(cls) -> SchemaVersion:
        """当前版本."""
        return cls.V1_0_0

    @classmethod
    def is_compatible(cls, version: str) -> bool:
        """检查版本是否向后兼容.

        策略:
        - 同 major 版本 → 兼容 (minor/patch 差异可忽略).
        - 不同 major → 不兼容.
        """
        try:
            major = int(version.split(".", maxsplit=1)[0])
        except (ValueError, IndexError):
            return False
        current_major = int(SCHEMA_VERSION.split(".", maxsplit=1)[0])
        return major == current_major


def validate_schema(data: dict[str, Any]) -> list[str]:
    """验证 dict 是否符合 Schema.

    Returns:
        list[str] 错误列表 (空 = 合法).
    """
    errors: list[str] = []

    # 检查必需字段.
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"missing required field: {field}")

    # 检查 schema_version (如果存在).
    if "schema_version" in data:
        version = str(data["schema_version"])
        if not SchemaVersion.is_compatible(version):
            errors.append(f"incompatible schema_version: {version} (current: {SCHEMA_VERSION})")

    # 检查类型.
    if "scene_id" in data and not isinstance(data["scene_id"], str):
        errors.append("scene_id must be str")
    if "objects" in data and not isinstance(data["objects"], list):
        errors.append("objects must be list")
    if "relations" in data and not isinstance(data["relations"], list):
        errors.append("relations must be list")
    if "regions" in data and not isinstance(data["regions"], list):
        errors.append("regions must be list")
    if "timestamp" in data and not isinstance(data["timestamp"], int | float):
        errors.append("timestamp must be number")

    return errors


def create_empty_schema(scene_id: str = "default") -> dict[str, Any]:
    """创建空 Schema (所有字段填默认值).

    用于向后兼容: 旧版本缺失字段填默认值.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "coordinate_system": COORDINATE_SYSTEM,
        "camera_pose": None,
        "objects": [],
        "relations": [],
        "regions": [],
        "occupancy": {},
        "memory": {"objects": []},
        "confidence": 0.0,
        "timestamp": 0.0,
    }
