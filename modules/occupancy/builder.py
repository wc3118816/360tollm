"""M018 Occupancy Map: 点云 → 占用栅格.

算法:
1. 点云点 → occupied 体素.
2. 相机位置 (sensor_origin) → 每个点的射线 → 途经体素标记 free.
   (3D Bresenham 射线投射)

用法:
    builder = OccupancyBuilder(OccupancyBuilderConfig(voxel_size=0.1))
    grid = builder.build(points, sensor_origin)
    status = grid.status_at([1.0, 2.0, 3.0])
    grid.decay(0.95)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.logging import get_logger

from .types import (
    _DEFAULT_CONF_THRESHOLD,
    _DEFAULT_DECAY_RATE,
    _DEFAULT_INIT_CONF,
    FREE,
    OCCUPIED,
    OccupancyGrid,
)

_LOG = get_logger("modules.occupancy.builder")

# 默认体素边长 (米).
_DEFAULT_VOXEL_SIZE = 0.1
# 默认 grid 边界扩展 (米, 在点云 bbox 外扩展).
_DEFAULT_PADDING = 1.0
# 射线最大长度 (米), 超过则不标记 free.
_DEFAULT_MAX_RAY_LENGTH = 50.0
# 点云维度常量.
_POINTS_NDIM = 2
_POINTS_DIM = 3
_ORIGIN_DIM = 3


@dataclass
class OccupancyBuilderConfig:
    """占用栅格构建配置."""

    voxel_size: float = _DEFAULT_VOXEL_SIZE
    padding: float = _DEFAULT_PADDING  # bbox 外扩展
    max_ray_length: float = _DEFAULT_MAX_RAY_LENGTH  # 射线最大长度
    init_confidence: float = _DEFAULT_INIT_CONF


class OccupancyBuilder:
    """点云 → 占用栅格.

    用法:
        builder = OccupancyBuilder()
        grid = builder.build(points, sensor_origin=np.zeros(3))
    """

    def __init__(self, cfg: OccupancyBuilderConfig | None = None) -> None:
        self.cfg = cfg or OccupancyBuilderConfig()
        self._log = _LOG

    def build(
        self,
        points: np.ndarray,
        sensor_origin: np.ndarray,
        existing: OccupancyGrid | None = None,
    ) -> OccupancyGrid:
        """构建/更新占用栅格.

        Args:
            points: (N, 3) float32 点云点 (世界坐标系).
            sensor_origin: (3,) 相机位置 (世界坐标系).
            existing: 已有栅格 (用于增量更新); None 则新建.

        Returns:
            OccupancyGrid.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != _POINTS_NDIM or pts.shape[1] != _POINTS_DIM:
            raise ValueError(f"points must be (N,3), got {pts.shape}")
        origin = np.asarray(sensor_origin, dtype=np.float64).ravel()
        if origin.shape[0] != _ORIGIN_DIM:
            raise ValueError(f"sensor_origin must be (3,), got {origin.shape}")

        if existing is not None:
            return self._update(existing, pts, origin)

        # 1. 计算 bbox + padding → grid 尺寸.
        if pts.shape[0] == 0:
            # 空点云 → 1x1x1 grid.
            min_bound = origin - self.cfg.padding
            max_bound = origin + self.cfg.padding
        else:
            min_bound = pts.min(axis=0) - self.cfg.padding
            max_bound = pts.max(axis=0) + self.cfg.padding
            # 确保 sensor_origin 在 grid 内.
            min_bound = np.minimum(min_bound, origin - self.cfg.padding)
            max_bound = np.maximum(max_bound, origin + self.cfg.padding)

        vs = self.cfg.voxel_size
        grid_shape = np.maximum(((max_bound - min_bound) / vs).astype(np.int64), 1)
        # 限制 grid 大小 (防止爆内存).
        max_dim = 200  # 每维最多 200 体素.
        grid_shape = np.minimum(grid_shape, max_dim)

        grid = np.zeros(tuple(grid_shape), dtype=np.int8)  # 全 UNKNOWN.
        confidence = np.zeros(tuple(grid_shape), dtype=np.float32)
        origin_grid = min_bound.copy()

        self._log.debug(
            "occupancy grid initialized",
            shape=tuple(grid_shape),
            origin=origin_grid.tolist(),
            voxel_size=vs,
        )

        grid_obj = OccupancyGrid(
            grid=grid,
            confidence=confidence,
            origin=origin_grid,
            voxel_size=vs,
            metadata={"n_updates": 0},
        )
        return self._update(grid_obj, pts, origin)

    def _update(
        self, grid: OccupancyGrid, points: np.ndarray, sensor_origin: np.ndarray
    ) -> OccupancyGrid:
        """更新栅格: 标记 occupied + 射线投射 free."""
        conf = self.cfg.init_confidence

        # 1. 标记点云点为 occupied.
        if points.shape[0] > 0:
            voxel_idx = grid.world_to_voxel(points)
            valid_mask = np.all(voxel_idx >= 0, axis=1)
            valid_idx = voxel_idx[valid_mask]
            # 去重.
            unique_idx = np.unique(valid_idx, axis=0)
            for idx in unique_idx:
                i0, i1, i2 = int(idx[0]), int(idx[1]), int(idx[2])
                # 只覆盖 UNKNOWN 和 FREE (不覆盖已有 occupied, 除非是更新).
                grid.grid[i0, i1, i2] = OCCUPIED
                grid.confidence[i0, i1, i2] = conf

        # 2. 射线投射: sensor_origin → 每个点, 途经体素标记 free.
        if points.shape[0] > 0:
            voxel_idx = grid.world_to_voxel(points)
            valid_mask = np.all(voxel_idx >= 0, axis=1)
            valid_points = points[valid_mask]
            valid_idx = voxel_idx[valid_mask]
            sensor_voxel = grid.world_to_voxel(sensor_origin)

            if np.all(sensor_voxel >= 0):
                n_rays = min(len(valid_points), 5000)  # 限制射线数防卡.
                step = max(1, len(valid_points) // n_rays) if n_rays > 0 else 1
                for i in range(0, len(valid_points), step):
                    end_voxel = valid_idx[i]
                    if np.all(end_voxel >= 0):
                        self._raycast_free(grid, sensor_voxel, end_voxel, conf)

        grid.metadata["n_updates"] = grid.metadata.get("n_updates", 0) + 1
        self._log.info(
            "occupancy grid updated",
            n_points=int(points.shape[0]),
            n_updates=grid.metadata["n_updates"],
        )
        return grid

    def _raycast_free(
        self,
        grid: OccupancyGrid,
        start: np.ndarray,
        end: np.ndarray,
        conf: float,
    ) -> None:
        """3D Bresenham 射线投射: 标记 start→end 途经的体素为 FREE (不覆盖 occupied).

        Args:
            grid: OccupancyGrid.
            start: (3,) int 起点体素索引.
            end: (3,) int 终点体素索引.
            conf: 置信度.
        """
        sx, sy, sz = int(start[0]), int(start[1]), int(start[2])
        ex, ey, ez = int(end[0]), int(end[1]), int(end[2])
        dx = abs(ex - sx)
        dy = abs(ey - sy)
        dz = abs(ez - sz)
        step_x = 1 if ex >= sx else -1
        step_y = 1 if ey >= sy else -1
        step_z = 1 if ez >= sz else -1

        # 3D Bresenham (简化版).
        xs, ys, zs = sx, sy, sz
        # 错误累加.
        err_1 = 2 * dx - dy - dz if dx >= max(dy, dz) else 0
        err_2 = 2 * dy - dx - dz if dy >= max(dx, dz) else 0
        err_3 = 2 * dz - dx - dy if dz >= max(dx, dy) else 0

        max_steps = dx + dy + dz + 1
        for _ in range(max_steps):
            # 跳过终点 (终点是 occupied).
            if xs == ex and ys == ey and zs == ez:
                break
            # 标记 FREE (不覆盖 OCCUPIED).
            if (
                0 <= xs < grid.shape[0]
                and 0 <= ys < grid.shape[1]
                and 0 <= zs < grid.shape[2]
                and grid.grid[xs, ys, zs] != OCCUPIED
            ):
                grid.grid[xs, ys, zs] = FREE
                grid.confidence[xs, ys, zs] = conf
            # 推进.
            if err_1 > 0:
                xs += step_x
                err_1 -= 2 * (dy + dz)
            if err_2 > 0:
                ys += step_y
                err_2 -= 2 * (dx + dz)
            if err_3 > 0:
                zs += step_z
                err_3 -= 2 * (dx + dy)
            err_1 += 2 * dx
            err_2 += 2 * dy
            err_3 += 2 * dz

    def decay_grid(
        self,
        grid: OccupancyGrid,
        decay_rate: float = _DEFAULT_DECAY_RATE,
        conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
    ) -> OccupancyGrid:
        """时间衰减封装."""
        grid.decay(decay_rate=decay_rate, conf_threshold=conf_threshold)
        return grid
