"""M015 点云生成器: depth + pose → 3D 点云.

两种投影模式:
1. perspective: 透视图 + 内参 K (针孔模型).
    每个像素 (u, v) 的射线方向 = K^-1 @ [u, v, 1].
    3D 点 (camera) = depth * ray.
2. equirectangular: 全景图 (M007 spherical 模型).
    每个像素 (u, v) 的射线 = pixel_to_ray(u, v, W, H).
    3D 点 (camera) = depth * ray.

世界坐标变换:
    point_world = R^T @ (point_camera - t) = pose.R.T @ point_camera - pose.R.T @ pose.t
    等价于 point_world = (pose.matrix^-1)[:3, :3] @ point_camera + (pose.matrix^-1)[:3, 3]
    (因为 matrix 是 world_to_camera, 其逆是 camera_to_world)

多帧融合:
    对每帧 (image, depth, pose) 生成局部点云 → 变换到 world → 合并.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modules.depth.types import DepthMap
from modules.logging import get_logger
from modules.projection import pixel_to_ray
from modules.vo.types import Pose

from .types import PointCloud

_LOG = get_logger("modules.pointcloud")

# 深度无效值阈值 (米).
_INVALID_DEPTH = 0.0
# 深度有效上限 (米), 过远的点视为噪声.
_DEFAULT_MAX_DEPTH = 100.0
# 降采样步长 (每隔 step 个像素取一个点).
_DEFAULT_STEP = 1
# 相机内参维度.
_K_SIZE = 3
# 彩色图像维度.
_NDIM_COLOR = 3


@dataclass
class PointCloudGeneratorConfig:
    """点云生成配置."""

    projection: Literal["perspective", "equirect"] = "equirect"
    max_depth: float = _DEFAULT_MAX_DEPTH  # 超过此深度视为噪声
    step: int = _DEFAULT_STEP  # 降采样步长 (1=全像素, 2=每隔一个)
    min_depth: float = 0.1  # 小于此深度视为噪声
    filter_nan: bool = True  # 移除 NaN/Inf


class PointCloudGenerator:
    """点云生成器.

    用法:
        gen = PointCloudGenerator(cfg)
        pc = gen.generate(image, depth_map, pose)  # 单帧
        pc = gen.generate_multi(frames)              # 多帧融合
    """

    def __init__(self, cfg: PointCloudGeneratorConfig | None = None) -> None:
        self.cfg = cfg or PointCloudGeneratorConfig()
        self._log = _LOG.bind(projection=self.cfg.projection)

    def generate(
        self,
        image: np.ndarray | None,
        depth_map: DepthMap,
        pose: Pose,
        camera_matrix: np.ndarray | None = None,
    ) -> PointCloud:
        """单帧 → 点云 (世界坐标系).

        Args:
            image: RGB uint8 (H, W, 3), 可选 (None 则无颜色).
            depth_map: DepthMap (M009).
            pose: Pose (M011), world_to_camera.
            camera_matrix: 3x3 内参 K. perspective 模式必需; equirect 忽略.

        Returns:
            PointCloud (世界坐标系).
        """
        H, W = depth_map.height, depth_map.width
        step = self.cfg.step

        # 采样像素网格 (降采样).
        v_coords, u_coords = np.mgrid[0:H:step, 0:W:step]
        u_flat = u_coords.ravel().astype(np.float64)
        v_flat = v_coords.ravel().astype(np.float64)
        depth_flat = depth_map.depth[::step, ::step].ravel().astype(np.float64)

        # 有效深度 mask.
        valid = (
            (depth_flat > self.cfg.min_depth)
            & (depth_flat < self.cfg.max_depth)
            & np.isfinite(depth_flat)
        )
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]
        d_valid = depth_flat[valid]

        # 计算 camera 坐标系下的 3D 点.
        if self.cfg.projection == "perspective":
            if camera_matrix is None:
                raise ValueError("perspective mode requires camera_matrix K")
            K = np.asarray(camera_matrix, dtype=np.float64)
            if K.shape != (_K_SIZE, _K_SIZE):
                raise ValueError(f"camera_matrix must be (3,3), got {K.shape}")
            # ray = K^-1 @ [u, v, 1]^T, point_cam = depth * ray
            K_inv = np.linalg.inv(K)
            ones = np.ones_like(u_valid)
            uv1 = np.stack([u_valid, v_valid, ones], axis=0)  # (3, N)
            rays = K_inv @ uv1  # (3, N)
            points_cam = d_valid * rays  # (3, N) broadcast
            points_cam = points_cam.T  # (N, 3)
        else:  # equirect
            # 每像素 → ray (M007), point_cam = depth * ray
            rays = np.array(
                [pixel_to_ray(u, v, W, H) for u, v in zip(u_valid, v_valid, strict=True)]
            )  # (N, 3)
            points_cam = d_valid[:, None] * rays  # (N, 3)

        # 变换到世界坐标系.
        # point_world = R^T @ (point_cam - t) = R^T @ point_cam - R^T @ t
        R = pose.R
        t = pose.t
        points_world = (points_cam - t) @ R  # (N, 3) @ (3,3) → 等价 R^T @ point_cam.T - ...

        # 提取颜色.
        colors = np.zeros((0, 3), dtype=np.uint8)
        if image is not None:
            img = np.asarray(image)
            if img.ndim == _NDIM_COLOR:
                # 采样对应像素的颜色.
                c_u = u_valid.astype(np.int32)
                c_v = v_valid.astype(np.int32)
                c_u = np.clip(c_u, 0, img.shape[1] - 1)
                c_v = np.clip(c_v, 0, img.shape[0] - 1)
                colors = img[c_v, c_u].astype(np.uint8)

        # NaN 过滤.
        if self.cfg.filter_nan:
            mask = np.isfinite(points_world).all(axis=1)
            points_world = points_world[mask]
            if colors.shape[0] > 0:
                colors = colors[mask]

        self._log.debug(
            "point cloud generated",
            n_points=points_world.shape[0],
            projection=self.cfg.projection,
            frame_id=pose.frame_id,
        )

        return PointCloud(
            points=points_world.astype(np.float32),
            colors=colors,
            metadata={
                "projection": self.cfg.projection,
                "source_frame_id": pose.frame_id,
                "source_timestamp": pose.timestamp,
                "depth_backend": depth_map.backend,
                "pose_backend": pose.metadata.get("backend", "unknown"),
                "image_shape": [H, W],
                "step": step,
            },
        )

    def generate_multi(
        self,
        frames: Sequence[Any],
        camera_matrix: np.ndarray | None = None,
    ) -> PointCloud:
        """多帧融合 → 单一点云 (世界坐标系).

        Args:
            frames: 帧序列, 每帧需有 .image / .depth / .pose.
                    depth: DepthMap 或 np.ndarray (H,W).
                    pose: Pose 或 4x4 matrix.
            camera_matrix: 内参 K (perspective 模式必需).

        Returns:
            合并后的 PointCloud.
        """
        if len(frames) == 0:
            return PointCloud(points=np.zeros((0, 3), dtype=np.float32))

        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        frame_ids: list[int] = []
        for frame in frames:
            # 提取 depth / pose / image.
            depth_attr = getattr(frame, "depth", None)
            pose_attr = getattr(frame, "pose", None)
            image_attr = getattr(frame, "image", None)

            if depth_attr is None or pose_attr is None:
                continue

            # 统一 DepthMap 类型.
            if isinstance(depth_attr, DepthMap):
                dm = depth_attr
            else:
                # 假设是 (H,W) 数组.
                dm = DepthMap(
                    depth=np.asarray(depth_attr, dtype=np.float32),
                    confidence=np.ones_like(depth_attr, dtype=np.float32),
                )

            # 统一 Pose 类型.
            if isinstance(pose_attr, Pose):
                pose = pose_attr
            else:
                pose = Pose(matrix=np.asarray(pose_attr, dtype=np.float64))

            pc = self.generate(image_attr, dm, pose, camera_matrix)
            all_points.append(pc.points)
            if pc.has_colors:
                all_colors.append(pc.colors)
            frame_ids.append(pose.frame_id)

        if not all_points:
            return PointCloud(points=np.zeros((0, 3), dtype=np.float32))

        points = np.vstack(all_points)
        colors = np.vstack(all_colors) if all_colors else np.zeros((0, 3), dtype=np.uint8)

        self._log.info(
            "multi-frame point cloud merged",
            n_frames=len(frame_ids),
            n_points=points.shape[0],
        )

        return PointCloud(
            points=points.astype(np.float32),
            colors=colors,
            metadata={
                "n_frames": len(frame_ids),
                "source_frame_ids": frame_ids,
                "projection": self.cfg.projection,
            },
        )


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """体素降采样.

    将点云体素化, 每个体素内取一个代表点 (均值).

    Args:
        points: (N, 3) float32.
        voxel_size: 体素边长 (米).
        colors: (N, 3) uint8, 可选.

    Returns:
        (downsampled_points, downsampled_colors) — colors 全零如果输入 colors=None.
    """
    if points.shape[0] == 0:
        return points, np.zeros((0, 3), dtype=np.uint8)

    # 体素索引.
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    # 唯一体素.
    unique_voxels, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)

    # 每个体素内取均值.
    n_voxels = unique_voxels.shape[0]
    down_points = np.zeros((n_voxels, 3), dtype=np.float32)
    down_colors = np.zeros((n_voxels, 3), dtype=np.uint8)
    for i in range(n_voxels):
        mask = inverse == i
        down_points[i] = points[mask].mean(axis=0)
        if colors is not None and colors.shape[0] > 0:
            down_colors[i] = colors[mask].mean(axis=0)
    return down_points, down_colors
