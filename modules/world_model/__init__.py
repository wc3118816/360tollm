"""M029 World Model API 子包.

公开 API:
    WorldModel - 统一世界模型 (坐标变换 + 物体查询)

坐标变换:
    WorldModel.world_to_camera(p_world, pose) → p_camera
    WorldModel.camera_to_world(p_camera, pose) → p_world

物体查询:
    model.object_pose(object_id)         → Object3D
    model.object_in_world(object_id)     → np.ndarray (3,)
    model.object_in_camera(object_id, pose) → np.ndarray (3,)
    model.object_distance(a, b)          → float (米)
    model.object_direction(from, to)     → np.ndarray (3,) 单位向量
    model.distance_to_camera(object_id)   → float (米)
    model.direction_from_camera(object_id) → np.ndarray (3,)

单位: SI (米, 弧度, 秒).

用法:
    from modules.world_model import WorldModel
    model = WorldModel()
    model.add_objects(object_list)
    p_cam = model.world_to_camera(p_world, pose)
    p_world_back = model.camera_to_world(p_cam, pose)
"""

from __future__ import annotations

from .api import WorldModel

__all__ = ["WorldModel"]
