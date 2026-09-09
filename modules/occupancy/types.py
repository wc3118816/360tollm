"""M018 Occupancy Map: 三态占用栅格.

三态:
- UNKNOWN (0): 未观测区域.
- FREE (1):    已观测且空闲.
- OCCUPIED (2): 已观测且占用 (有点云点).

数据结构:
- grid: int8 (Gx, Gy, Gz) 三态栅格.
- origin: 栅格原点在世界坐标系的位置 (x0, y0, z0).
- voxel_size: 体素边长 (米).
- confidence: float32 (Gx, Gy, Gz) 置信度 (0-1), 随时间衰减.

时间衰减:
- decay(): confidence *= decay_rate, 低于阈值则 → UNKNOWN.
- update(): 新观测会重置 confidence.

查询:
- status_at(world_pos): 返回三态.
- world_to_voxel(world_pos) / voxel_to_world(voxel_idx).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 三态枚举 (用 int8).
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

_NDIM_3D = 3
_NDIM_2D = 2
# 默认衰减率 (每帧 confidence *= decay_rate).
_DEFAULT_DECAY_RATE = 0.95
# 默认 confidence 阈值, 低于此值 → UNKNOWN.
_DEFAULT_CONF_THRESHOLD = 0.1
# 默认初始 confidence.
_DEFAULT_INIT_CONF = 1.0


@dataclass(slots=True)
class OccupancyGrid:
    """三态占用栅格.

    Attributes:
        grid: int8 (Gx, Gy, Gz) 三态栅格 (UNKNOWN/FREE/OCCUPIED).
        confidence: float32 (Gx, Gy, Gz) 置信度, 随时间衰减.
        origin: float64 (3,) 栅格原点在世界坐标系的位置.
        voxel_size: float 体素边长 (米).
        metadata: 扩展字段 (n_updates / decay_count 等).
    """

    grid: np.ndarray
    confidence: np.ndarray
    origin: np.ndarray
    voxel_size: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=np.int8)
        if self.grid.ndim != _NDIM_3D:
            raise ValueError(f"grid must be 3D, got {self.grid.ndim}D")
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        if self.confidence.shape != self.grid.shape:
            raise ValueError(
                f"confidence shape {self.confidence.shape} != grid shape {self.grid.shape}"
            )
        self.origin = np.asarray(self.origin, dtype=np.float64).ravel()
        if self.origin.shape[0] != _NDIM_3D:
            raise ValueError(f"origin must be (3,), got {self.origin.shape}")
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {self.voxel_size}")

    @property
    def shape(self) -> tuple[int, int, int]:
        """栅格形状 (Gx, Gy, Gz)."""
        return self.grid.shape  # type: ignore[return-value]

    @property
    def n_voxels(self) -> int:
        return self.grid.size

    def world_to_voxel(self, world_pos: np.ndarray) -> np.ndarray:
        """世界坐标 → 体素索引 (四舍五入).

        Args:
            world_pos: (3,) or (N, 3) float.

        Returns:
            voxel_idx: (3,) int or (N, 3) int. 超出范围的索引为 -1.
        """
        pts = np.asarray(world_pos, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts.reshape(1, -1)
        idx = np.floor((pts - self.origin) / self.voxel_size).astype(np.int64)
        # 范围检查.
        gx, gy, gz = self.shape
        valid = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < gx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < gy)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < gz)
        )
        idx[~valid] = -1
        if single:
            return idx[0]
        return idx

    def voxel_to_world(self, voxel_idx: np.ndarray) -> np.ndarray:
        """体素索引 → 世界坐标 (体素中心).

        Args:
            voxel_idx: (3,) or (N, 3) int.

        Returns:
            world_pos: (3,) or (N, 3) float64.
        """
        idx = np.asarray(voxel_idx, dtype=np.int64)
        single = idx.ndim == 1
        if single:
            idx = idx.reshape(1, -1)
        world = self.origin + (idx + 0.5) * self.voxel_size
        if single:
            return world[0]
        return world

    def status_at(self, world_pos: np.ndarray) -> int:
        """查询世界坐标的占用状态.

        Returns:
            UNKNOWN (0) / FREE (1) / OCCUPIED (2).
            超出范围返回 UNKNOWN.
        """
        idx = self.world_to_voxel(world_pos)
        if np.any(idx < 0):
            return UNKNOWN
        return int(self.grid[idx[0], idx[1], idx[2]])

    def is_occupied(self, world_pos: np.ndarray) -> bool:
        return self.status_at(world_pos) == OCCUPIED

    def is_free(self, world_pos: np.ndarray) -> bool:
        return self.status_at(world_pos) == FREE

    def is_unknown(self, world_pos: np.ndarray) -> bool:
        return self.status_at(world_pos) == UNKNOWN

    def decay(
        self,
        decay_rate: float = _DEFAULT_DECAY_RATE,
        conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
    ) -> None:
        """时间衰减: confidence *= decay_rate, 低于阈值 → UNKNOWN.

        Args:
            decay_rate: 衰减率 (0-1).
            conf_threshold: 低于此值 → UNKNOWN.
        """
        self.confidence *= decay_rate
        # 低置信度 → UNKNOWN.
        low_mask = self.confidence < conf_threshold
        self.grid[low_mask] = UNKNOWN
        self.metadata["decay_count"] = self.metadata.get("decay_count", 0) + 1

    def summary(self) -> dict[str, Any]:
        """统计摘要."""
        n_unknown = int(np.sum(self.grid == UNKNOWN))
        n_free = int(np.sum(self.grid == FREE))
        n_occupied = int(np.sum(self.grid == OCCUPIED))
        return {
            "shape": list(self.shape),
            "voxel_size": self.voxel_size,
            "origin": self.origin.tolist(),
            "n_unknown": n_unknown,
            "n_free": n_free,
            "n_occupied": n_occupied,
            "n_voxels": self.n_voxels,
            "mean_confidence": float(self.confidence.mean()),
            "metadata": self.metadata,
        }
