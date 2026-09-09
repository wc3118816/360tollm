"""M015/M016 点云生成与优化子包.

公开 API:
    PointCloud              - 点云数据结构 (points + colors + normals + PLY)
    PointCloudGenerator     - 生成器 (depth + pose → 点云) [M015]
    PointCloudGeneratorConfig - 生成器配置 [M015]
    PointCloudOptimizer     - 优化器 (法向量 + 滤波) [M016]
    OptimizerConfig         - 优化器配置 [M016]
    voxel_downsample        - 体素降采样 [M015]
    estimate_normals       - 法向量估计 (PCA) [M016]
    estimate_normals_pc    - PointCloud 法向量 [M016]
    statistical_outlier_removal - 统计滤波 [M016]
    radius_outlier_removal      - 半径滤波 [M016]
    filter_statistical_pc / filter_radius_pc - PointCloud 滤波 [M016]

用法:
    from modules.pointcloud import PointCloudGenerator, PointCloudOptimizer
    gen = PointCloudGenerator()
    pc = gen.generate(image, depth_map, pose)
    opt = PointCloudOptimizer()
    opt_pc = opt.optimize(pc)
    opt_pc.save_ply("output.ply")
"""

from __future__ import annotations

from .filter import (
    filter_radius_pc,
    filter_statistical_pc,
    radius_outlier_removal,
    statistical_outlier_removal,
)
from .generator import PointCloudGenerator, PointCloudGeneratorConfig, voxel_downsample
from .normals import estimate_normals, estimate_normals_pc
from .optimizer import OptimizerConfig, PointCloudOptimizer
from .types import PointCloud

__all__ = [
    # types
    "PointCloud",
    # generator
    "PointCloudGenerator",
    "PointCloudGeneratorConfig",
    "voxel_downsample",
    # optimizer
    "PointCloudOptimizer",
    "OptimizerConfig",
    # normals
    "estimate_normals",
    "estimate_normals_pc",
    # filter
    "statistical_outlier_removal",
    "radius_outlier_removal",
    "filter_statistical_pc",
    "filter_radius_pc",
]
