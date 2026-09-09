"""M021 2D→3D 目标关联: 2D Detection + 深度图 + Pose → 3D Object.

算法:
1. 对每个 2D Detection 的 bbox, 取中心像素 (cx, cy).
2. 从深度图取该像素的深度值 d (取 bbox 内中位数, 抗噪).
3. 用投影模型 (equirect/perspective) 把像素反投影为相机坐标射线 ray.
4. point_cam = d * ray.
5. point_world = (point_cam - t) @ R  (Pose 变换).
6. 估计 3D 尺寸: 从 bbox 内深度差和像素范围估算 dx/dy/dz.

输出:
- Object3D: center (世界坐标) + bbox_3d (8 顶点) + size_3d + confidence.
- object_id: 自动生成 "{label}_{frame_id:03d}_{idx:02d}".

AC 验证:
- 静态对象连续帧位置稳定 (同一物体 world center 方差小).
- 坐标单位统一 (米).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from modules.depth.types import DepthMap
from modules.detection import Detection, DetectionList
from modules.logging import get_logger
from modules.projection import pixel_to_ray
from modules.vo.types import Pose

from .types import Object3D, Object3DList

_LOG = get_logger("modules.objects.associator")

# 内参维度.
_K_SIZE = 3
# 3D 维度.
_NDIM_3D = 3
# bbox_3d 顶点数.
_BBOX_3D_VERTS = 8
# 深度无效阈值.
_MIN_DEPTH = 0.1
_MAX_DEPTH = 100.0
# 深度采样: 取 bbox 内中位数 (抗噪).
_DEPTH_SAMPLE_MODE = "median"


@dataclass
class AssociatorConfig:
    """关联器配置."""

    projection: Literal["perspective", "equirect"] = "equirect"
    min_depth: float = _MIN_DEPTH
    max_depth: float = _MAX_DEPTH
    # object_id 前缀 (None 用 label).
    id_prefix: str | None = None
    # 深度采样模式: "median" (bbox 内中位数) 或 "center" (中心像素).
    depth_sample: str = _DEPTH_SAMPLE_MODE


class ObjectAssociator:
    """2D Detection + DepthMap + Pose → Object3DList.

    用法:
        assoc = ObjectAssociator(AssociatorConfig(projection="equirect"))
        obj_list = assoc.associate(detection_list, depth_map, pose)
    """

    def __init__(self, cfg: AssociatorConfig | None = None) -> None:
        self.cfg = cfg or AssociatorConfig()
        self._log = _LOG.bind(projection=self.cfg.projection)
        self._id_counter: dict[str, int] = {}  # label → count

    def associate(
        self,
        detections: DetectionList,
        depth_map: DepthMap,
        pose: Pose,
        camera_matrix: np.ndarray | None = None,
    ) -> Object3DList:
        """2D Detection + 深度图 + Pose → 3D 物体列表.

        Args:
            detections: 2D 检测结果.
            depth_map: 深度图 (M009).
            pose: 位姿 (M011, world_to_camera).
            camera_matrix: 3x3 内参 K (perspective 模式必需).

        Returns:
            Object3DList.
        """
        H, W = depth_map.height, depth_map.width
        objects: list[Object3D] = []

        for idx, det in enumerate(detections.detections):
            obj = self._associate_one(
                det, depth_map, pose, H, W, camera_matrix, idx, detections.timestamp
            )
            if obj is not None:
                objects.append(obj)

        self._log.info(
            "association done",
            n_detections=detections.n_detections,
            n_objects=len(objects),
            frame_id=pose.frame_id,
        )

        return Object3DList(
            objects=objects,
            frame_id=pose.frame_id,
            timestamp=detections.timestamp,
            metadata={
                "projection": self.cfg.projection,
                "n_detections": detections.n_detections,
                "n_objects": len(objects),
            },
        )

    def _associate_one(  # noqa: PLR0912, PLR0915, PLR0917
        self,
        det: Detection,
        depth_map: DepthMap,
        pose: Pose,
        H: int,
        W: int,
        camera_matrix: np.ndarray | None,
        idx: int,
        timestamp: float,
    ) -> Object3D | None:
        """单个 Detection → Object3D."""
        x_min, y_min, x_max, y_max = det.bbox.tolist()
        # 裁剪到图像范围.
        x_min = max(0, min(x_min, W - 1))
        x_max = max(0, min(x_max, W - 1))
        y_min = max(0, min(y_min, H - 1))
        y_max = max(0, min(y_max, H - 1))
        if x_max <= x_min or y_max <= y_min:
            return None

        # 1. 取深度值.
        depth_region = depth_map.depth[y_min : y_max + 1, x_min : x_max + 1]
        if self.cfg.depth_sample == "center":
            cx_pix = (x_min + x_max) // 2
            cy_pix = (y_min + y_max) // 2
            d = float(depth_map.depth[cy_pix, cx_pix])
        else:
            valid = (depth_region > self.cfg.min_depth) & (depth_region < self.cfg.max_depth)
            if not np.any(valid):
                return None
            d = float(np.median(depth_region[valid]))

        if d < self.cfg.min_depth or d > self.cfg.max_depth or not np.isfinite(d):
            return None

        # 2. 像素 → 相机坐标射线.
        cx_pix = (x_min + x_max) / 2.0
        cy_pix = (y_min + y_max) / 2.0
        if self.cfg.projection == "perspective":
            if camera_matrix is None:
                raise ValueError("perspective mode requires camera_matrix K")
            K = np.asarray(camera_matrix, dtype=np.float64)
            if K.shape != (_K_SIZE, _K_SIZE):
                raise ValueError(f"camera_matrix must be (3,3), got {K.shape}")
            K_inv = np.linalg.inv(K)
            uv1 = np.array([cx_pix, cy_pix, 1.0], dtype=np.float64)
            ray = K_inv @ uv1
            ray = ray / np.linalg.norm(ray)
        else:  # equirect
            ray = np.asarray(pixel_to_ray(cx_pix, cy_pix, W, H), dtype=np.float64)

        # 3. 相机坐标.
        point_cam = d * ray  # (3,)

        # 4. 世界坐标.
        R = pose.R
        t = pose.t
        point_world = (point_cam - t) @ R  # (3,)

        # 5. 估算 3D 尺寸 (简化版).
        # dx/dy: 从 bbox 像素尺寸和深度估算.
        # dz: 从 bbox 内深度差估算.
        bw_pix = x_max - x_min
        bh_pix = y_max - y_min
        # 焦距近似 (perspective): f = K[0,0]; equirect: f = W / (2*pi).
        if self.cfg.projection == "perspective" and camera_matrix is not None:
            f_x = float(camera_matrix[0, 0])
            f_y = float(camera_matrix[1, 1])
        else:
            f_x = W / (2.0 * np.pi)
            f_y = f_x
        dx = abs(d * bw_pix / max(f_x, 1.0))
        dy = abs(d * bh_pix / max(f_y, 1.0))
        dz = float(np.ptp(depth_region[depth_region > 0])) if np.any(depth_region > 0) else 0.0
        if dz <= 0:
            dz = max(dx, dy) * 0.5  # 深度均匀时用默认值.
        size_3d = np.array([dx, dy, dz], dtype=np.float32)

        # 6. 3D bbox (8 顶点, 轴对齐).
        half = size_3d / 2.0
        corners = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float32,
            )
            * half
        )
        bbox_3d = point_world.astype(np.float32) + corners

        # 7. object_id.
        label = det.label
        prefix = self.cfg.id_prefix or label
        count = self._id_counter.get(label, 0)
        self._id_counter[label] = count + 1
        object_id = f"{prefix}_{pose.frame_id:03d}_{count:02d}"

        return Object3D(
            object_id=object_id,
            label=label,
            center=point_world.astype(np.float32),
            bbox_3d=bbox_3d,
            size_3d=size_3d,
            confidence=det.confidence,
            frame_id=pose.frame_id,
            timestamp=timestamp,
            metadata={
                "source_bbox_2d": [x_min, y_min, x_max, y_max],
                "depth_value": round(d, 4),
                "bbox_2d_area": int(bw_pix * bh_pix),
                "projection": self.cfg.projection,
            },
        )

    def reset_ids(self) -> None:
        """重置 object_id 计数器."""
        self._id_counter.clear()
