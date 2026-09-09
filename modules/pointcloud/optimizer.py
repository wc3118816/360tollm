"""M016 点云优化器: 整合法向量估计 + 滤波 + 体素降采样.

PointCloudOptimizer 链式流程:
    1. 体素降采样 (减少点数, M015 已实现)
    2. 统计滤波 (移除离群点)
    3. 半径滤波 (移除孤立点)
    4. 法向量估计 (PCA 局部拟合)

每步可选, 通过配置控制.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.logging import get_logger

from .filter import filter_radius_pc, filter_statistical_pc
from .generator import voxel_downsample
from .normals import estimate_normals_pc
from .types import PointCloud

_LOG = get_logger("modules.pointcloud.optimizer")


@dataclass
class OptimizerConfig:
    """点云优化配置."""

    # 体素降采样.
    voxel_size: float = 0.0  # 0 = 不降采样
    # 统计滤波.
    stat_k: int = 10
    stat_n_sigma: float = 2.0
    enable_stat_filter: bool = True
    # 半径滤波.
    radius: float = 0.1
    radius_min_neighbors: int = 5
    enable_radius_filter: bool = True
    # 法向量估计.
    normals_k: int = 10
    enable_normals: bool = True


class PointCloudOptimizer:
    """点云优化器.

    用法:
        opt = PointCloudOptimizer(OptimizerConfig(voxel_size=0.05))
        opt_pc = opt.optimize(pc)
    """

    def __init__(self, cfg: OptimizerConfig | None = None) -> None:
        self.cfg = cfg or OptimizerConfig()
        self._log = _LOG

    def optimize(self, pc: PointCloud) -> PointCloud:
        """链式优化点云."""
        original_n = pc.n_points
        current = pc  # 直接修改 (调用方需自行 copy)

        # 1. 体素降采样.
        if self.cfg.voxel_size > 0 and current.n_points > 0:
            down_points, down_colors = voxel_downsample(
                current.points, self.cfg.voxel_size, current.colors if current.has_colors else None
            )
            current = PointCloud(
                points=down_points,
                colors=down_colors,
                metadata=current.metadata,
            )
            self._log.debug(
                "voxel downsample",
                n_before=original_n,
                n_after=current.n_points,
                voxel_size=self.cfg.voxel_size,
            )

        # 2. 统计滤波.
        if self.cfg.enable_stat_filter and current.n_points > 0:
            current = filter_statistical_pc(
                current,
                k=self.cfg.stat_k,
                n_sigma=self.cfg.stat_n_sigma,
            )

        # 3. 半径滤波.
        if self.cfg.enable_radius_filter and current.n_points > 0:
            current = filter_radius_pc(
                current,
                radius=self.cfg.radius,
                min_neighbors=self.cfg.radius_min_neighbors,
            )

        # 4. 法向量估计.
        if self.cfg.enable_normals and current.n_points > 0:
            current = estimate_normals_pc(
                current,
                k=self.cfg.normals_k,
            )

        current.metadata["optimized"] = True
        current.metadata["n_before_optimize"] = original_n
        current.metadata["n_after_optimize"] = current.n_points
        self._log.info(
            "point cloud optimized",
            n_before=original_n,
            n_after=current.n_points,
            has_normals=current.has_normals,
        )
        return current
