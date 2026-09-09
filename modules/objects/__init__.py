"""M021 2D→3D 目标关联子包.

公开 API:
    Object3D              - 3D 物体数据结构 (center + bbox_3d + confidence)
    Object3DList          - 3D 物体列表 (单帧)
    ObjectAssociator      - 2D Detection + 深度 + Pose → 3D 物体
    AssociatorConfig      - 关联器配置

用法:
    from modules.objects import ObjectAssociator
    assoc = ObjectAssociator()
    obj_list = assoc.associate(detections, depth_map, pose)
    for obj in obj_list.objects:
        print(obj.object_id, obj.center)
"""

from __future__ import annotations

from .associator import AssociatorConfig, ObjectAssociator
from .types import Object3D, Object3DList

__all__ = [
    "Object3D",
    "Object3DList",
    "ObjectAssociator",
    "AssociatorConfig",
]
