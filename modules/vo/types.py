"""M011 Visual Odometry: Pose 数据结构.

Pose 用 4x4 SE3 矩阵表示 world_to_camera 变换:
    [R | t]   R: 3x3 旋转矩阵 (world → camera)
    [0 | 1]   t: 3x1 平移向量 (world 原点在 camera 坐标系的位置)

约定 (与 M007 一致):
    camera: +x=forward, +y=up, +z=right
    world:  初始与 camera 0 重合, 后续随相机运动

Trajectory: 有序 Pose 序列, 代表相机轨迹.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_NDIM_SQUARE = 2
_SE3_SIZE = 4
_NDIM_3D = 3
# 轨迹长度阈值 (少于则不计算 displacement/path_length).
_MIN_POSES_FOR_DISPLACEMENT = 2
# 灰度图通道数 (用于 ndim 判断).
_NDIM_GRAYSCALE = 2


@dataclass(slots=True)
class Pose:
    """单个相机位姿 (SE3).

    Attributes:
        matrix: 4x4 float64 SE3 矩阵 [R|t; 0|1] (world_to_camera).
        timestamp: 该 pose 对应的帧时间戳 (秒).
        frame_id: 对应的 Frame.frame_id.
        confidence: 位姿置信度 [0, 1]. RANSAC inlier ratio 或常数 1.0.
        metadata: 扩展字段 (特征匹配数 / inlier 数 / 求解方法 等).
    """

    matrix: np.ndarray
    timestamp: float = 0.0
    frame_id: int = 0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape != (_SE3_SIZE, _SE3_SIZE):
            raise ValueError(f"matrix must be (4,4), got {self.matrix.shape}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if not np.isfinite(self.matrix).all():
            raise ValueError("matrix contains NaN or Inf")

    @property
    def R(self) -> np.ndarray:
        """3x3 旋转矩阵 (world → camera)."""
        return self.matrix[:_NDIM_3D, :_NDIM_3D]

    @property
    def t(self) -> np.ndarray:
        """3x1 平移向量 (world 原点在 camera 坐标系的位置)."""
        return self.matrix[:_NDIM_3D, _NDIM_3D]

    @property
    def position(self) -> np.ndarray:
        """相机在 world 坐标系的位置 = -R^T @ t (3,)."""
        return -self.R.T @ self.t

    @property
    def yaw(self) -> float:
        """朝向角 (rad), 从 R 提取. 与 M007 约定一致."""
        # forward = R @ (1,0,0) → (R[0,0], R[1,0], R[2,0])
        # yaw = atan2(-z, x) = atan2(-R[2,0], R[0,0])
        return float(np.arctan2(-self.R[2, 0], self.R[0, 0]))

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容 dict."""
        return {
            "matrix": self.matrix.tolist(),
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "confidence": self.confidence,
            "position": self.position.tolist(),
            "yaw": self.yaw,
            "metadata": self.metadata,
        }

    @classmethod
    def identity(cls, timestamp: float = 0.0, frame_id: int = 0) -> Pose:
        """单位 pose (相机在 world 原点, 无旋转)."""
        return cls(
            matrix=np.eye(_SE3_SIZE, dtype=np.float64),
            timestamp=timestamp,
            frame_id=frame_id,
        )


@dataclass(slots=True)
class Trajectory:
    """有序 Pose 序列, 代表相机轨迹.

    Attributes:
        poses: 有序 Pose 列表 (按 timestamp/frame_id 递增).
        backend: VO 后端名 (dummy / orb / optical_flow).
        metadata: 扩展字段 (总帧数 / 失败数 / 总位移 等).
    """

    poses: list[Pose]
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.poses, list):
            raise TypeError(f"poses must be list, got {type(self.poses)}")

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self):
        return iter(self.poses)

    def __getitem__(self, idx: int) -> Pose:
        return self.poses[idx]

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) 位置数组 (world 坐标系)."""
        if not self.poses:
            return np.zeros((0, _NDIM_3D), dtype=np.float64)
        return np.array([p.position for p in self.poses], dtype=np.float64)

    @property
    def timestamps(self) -> np.ndarray:
        """(N,) 时间戳数组."""
        if not self.poses:
            return np.zeros((0,), dtype=np.float64)
        return np.array([p.timestamp for p in self.poses], dtype=np.float64)

    def has_nan(self) -> bool:
        """检查轨迹是否含 NaN (AC: 轨迹无 NaN)."""
        return any(not np.isfinite(p.matrix).all() for p in self.poses)

    def total_displacement(self) -> float:
        """起点到终点的直线距离 (米)."""
        if len(self.poses) < _MIN_POSES_FOR_DISPLACEMENT:
            return 0.0
        start = self.poses[0].position
        end = self.poses[-1].position
        return float(np.linalg.norm(end - start))

    def total_path_length(self) -> float:
        """累计路径长度 (米, 逐段欧氏距离)."""
        if len(self.poses) < _MIN_POSES_FOR_DISPLACEMENT:
            return 0.0
        positions = self.positions
        diffs = np.diff(positions, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def summary(self) -> dict[str, Any]:
        """统计摘要."""
        return {
            "backend": self.backend,
            "n_poses": len(self.poses),
            "has_nan": self.has_nan(),
            "total_displacement": round(self.total_displacement(), 4),
            "total_path_length": round(self.total_path_length(), 4),
            "start_pos": self.poses[0].position.tolist() if self.poses else [],
            "end_pos": self.poses[-1].position.tolist() if self.poses else [],
            "metadata": self.metadata,
        }
