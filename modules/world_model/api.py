"""M029 World Model API: 统一 World/Camera/Object 坐标系.

API:
- world_to_camera(p_world, pose) → p_camera
- camera_to_world(p_camera, pose) → p_world
- object_pose(object_id) → Object3D (位置+姿态)
- object_distance(a, b) → float (米)
- object_direction(from_obj, to_obj) → np.ndarray (3,) 单位向量
- object_in_world(object_id) → np.ndarray (3,) 世界坐标
- object_in_camera(object_id, pose) → np.ndarray (3,) 相机坐标

约定 (与 M011 Pose 一致):
- Pose.matrix = [R|t; 0|1] 是 world_to_camera 变换.
- R: world → camera 旋转.
- t: world 原点在 camera 坐标系的位置.
- camera: +x=forward, +y=up, +z=right.
- world: 初始与 camera 0 重合.

单位: SI (米, 弧度, 秒).

用法:
    model = WorldModel()
    model.add_objects(object_list)
    p_cam = model.world_to_camera(np.array([1,2,3]), pose)
    p_world = model.camera_to_world(p_cam, pose)
    d = model.object_distance("a", "b")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.logging import get_logger
from modules.objects import Object3D, Object3DList
from modules.vo.types import Pose

_LOG = get_logger("modules.world_model")

_NDIM_3D = 3
_SE3_SIZE = 4
# 坐标单位 (SI).
_UNIT_LENGTH = "meter"
_UNIT_ANGLE = "radian"
# 同位置判断阈值.
_SAME_POS_EPS = 1e-10


@dataclass
class WorldModel:
    """统一世界模型 API.

    管理物体 + 位姿 + 坐标变换.

    Attributes:
        objects: dict[str, Object3D] object_id → Object3D.
        current_pose: 当前相机位姿 (None = 世界原点).
        metadata: 模型级元数据.
    """

    objects: dict[str, Object3D] = field(default_factory=dict)
    current_pose: Pose | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._log = _LOG

    # === 物体管理 ===

    def add_objects(self, object_list: Object3DList) -> None:
        """从 Object3DList 添加物体 (重复 ID 覆盖)."""
        for obj in object_list.objects:
            self.objects[obj.object_id] = obj
        self._log.debug(
            "objects added",
            n_new=object_list.n_objects,
            n_total=len(self.objects),
        )

    def add_object(self, obj: Object3D) -> None:
        """添加单个物体."""
        self.objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> Object3D | None:
        """获取物体."""
        return self.objects.get(object_id)

    def object_pose(self, object_id: str) -> Object3D | None:
        """获取物体的位姿 (位置+尺寸, M029 API).

        Returns:
            Object3D 或 None (不存在).
        """
        return self.objects.get(object_id)

    def object_in_world(self, object_id: str) -> np.ndarray | None:
        """获取物体在世界坐标系的位置 (3,).

        Returns:
            float32 (3,) 或 None.
        """
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        return obj.center.copy()

    def object_in_camera(self, object_id: str, pose: Pose | None = None) -> np.ndarray | None:
        """获取物体在相机坐标系的位置 (3,).

        Args:
            object_id: 物体 ID.
            pose: 相机位姿 (None 用 current_pose).

        Returns:
            float32 (3,) 或 None.
        """
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        p = pose or self.current_pose
        if p is None:
            return obj.center.copy()  # 无 pose = world == camera.
        return self.world_to_camera(obj.center, p)

    # === 坐标变换 (AC: 双向可验证) ===

    @staticmethod
    def world_to_camera(p_world: np.ndarray, pose: Pose) -> np.ndarray:
        """世界坐标 → 相机坐标.

        p_camera = R @ p_world + t

        Args:
            p_world: (3,) or (N, 3) float 世界坐标 (米).
            pose: 相机位姿.

        Returns:
            float64 同形状, 相机坐标 (米).
        """
        pts = np.asarray(p_world, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts.reshape(1, -1)
        if pts.shape[1] != _NDIM_3D:
            raise ValueError(f"p_world must be (3,) or (N,3), got {pts.shape}")
        R = pose.R
        t = pose.t
        p_cam = (R @ pts.T).T + t  # (N, 3)
        if single:
            return p_cam[0]
        return p_cam

    @staticmethod
    def camera_to_world(p_camera: np.ndarray, pose: Pose) -> np.ndarray:
        """相机坐标 → 世界坐标.

        p_world = R^T @ (p_camera - t)

        Args:
            p_camera: (3,) or (N, 3) float 相机坐标 (米).
            pose: 相机位姿.

        Returns:
            float64 同形状, 世界坐标 (米).
        """
        pts = np.asarray(p_camera, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts.reshape(1, -1)
        if pts.shape[1] != _NDIM_3D:
            raise ValueError(f"p_camera must be (3,) or (N,3), got {pts.shape}")
        R = pose.R
        t = pose.t
        p_world = (R.T @ (pts - t).T).T  # (N, 3)
        if single:
            return p_world[0]
        return p_world

    # === 物体间关系 (M029 API) ===

    def object_distance(self, object_id_a: str, object_id_b: str) -> float | None:
        """两物体间的欧氏距离 (米, SI).

        Returns:
            float 或 None (物体不存在).
        """
        obj_a = self.objects.get(object_id_a)
        obj_b = self.objects.get(object_id_b)
        if obj_a is None or obj_b is None:
            return None
        return float(np.linalg.norm(obj_a.center - obj_b.center))

    def object_direction(self, from_object_id: str, to_object_id: str) -> np.ndarray | None:
        """从一个物体到另一个物体的方向 (单位向量, 世界坐标).

        Returns:
            float32 (3,) 单位向量 或 None.
        """
        obj_from = self.objects.get(from_object_id)
        obj_to = self.objects.get(to_object_id)
        if obj_from is None or obj_to is None:
            return None
        diff = obj_to.center - obj_from.center
        norm = float(np.linalg.norm(diff))
        if norm < _SAME_POS_EPS:
            return np.zeros(_NDIM_3D, dtype=np.float32)
        return (diff / norm).astype(np.float32)

    def distance_to_camera(self, object_id: str, pose: Pose | None = None) -> float | None:
        """物体到相机的距离 (米).

        Args:
            object_id: 物体 ID.
            pose: 相机位姿 (None 用 current_pose).

        Returns:
            float 或 None.
        """
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        p = pose or self.current_pose
        if p is None:
            return float(np.linalg.norm(obj.center))  # 相机在原点.
        cam_pos = p.position  # 相机在世界坐标系的位置.
        return float(np.linalg.norm(obj.center - cam_pos))

    def direction_from_camera(self, object_id: str, pose: Pose | None = None) -> np.ndarray | None:
        """物体相对相机的方向 (单位向量, 世界坐标).

        Returns:
            float32 (3,) 或 None.
        """
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        p = pose or self.current_pose
        cam_pos = np.zeros(_NDIM_3D, dtype=np.float64) if p is None else p.position
        diff = obj.center.astype(np.float64) - cam_pos
        norm = float(np.linalg.norm(diff))
        if norm < _SAME_POS_EPS:
            return np.zeros(_NDIM_3D, dtype=np.float32)
        return (diff / norm).astype(np.float32)

    # === 查询 ===

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def labels(self) -> list[str]:
        return list({o.label for o in self.objects.values()})

    def filter_by_label(self, label: str) -> list[Object3D]:
        """按类别过滤物体."""
        return [o for o in self.objects.values() if o.label == label]

    def find_nearest(self, object_id: str, label: str | None = None) -> tuple[str, float] | None:
        """找离指定物体最近的其他物体.

        Args:
            object_id: 参考物体 ID.
            label: 限定类别 (None 不限).

        Returns:
            (nearest_id, distance) 或 None.
        """
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        nearest_id: str | None = None
        nearest_dist = float("inf")
        for other_id, other in self.objects.items():
            if other_id == object_id:
                continue
            if label is not None and other.label != label:
                continue
            d = float(np.linalg.norm(obj.center - other.center))
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = other_id
        if nearest_id is None:
            return None
        return (nearest_id, nearest_dist)

    def summary(self) -> dict[str, Any]:
        """模型摘要."""
        return {
            "n_objects": self.n_objects,
            "labels": self.labels,
            "current_pose_set": self.current_pose is not None,
            "unit_length": _UNIT_LENGTH,
            "unit_angle": _UNIT_ANGLE,
            "metadata": self.metadata,
        }
