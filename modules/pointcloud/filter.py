"""M016 点云优化: 离群点滤波.

两种滤波:
1. 统计滤波 (statistical_outlier_removal):
   - 每个点到 k 近邻的平均距离.
   - 全局 mean + std.
   - 超过 mean + n_sigma * std 的点视为离群.
2. 半径滤波 (radius_outlier_removal):
   - 每个点在半径 r 内的邻点数.
   - 少于 min_neighbors 的点移除.

限制:
- 同样用 brute-force kNN (O(N^2)), 适合中小点云.
"""

from __future__ import annotations

import numpy as np

from modules.logging import get_logger

from .types import PointCloud

_LOG = get_logger("modules.pointcloud.filter")

# 默认统计滤波参数.
_DEFAULT_K = 10
_DEFAULT_N_SIGMA = 2.0
# 默认半径滤波参数.
_DEFAULT_RADIUS = 0.1
_DEFAULT_MIN_NEIGHBORS = 5
# 3D 维度.
_NDIM_3D = 3


def _pairwise_dists(points: np.ndarray) -> np.ndarray:
    """计算 (N, N) 距离矩阵 (brute-force)."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def statistical_outlier_removal(
    points: np.ndarray,
    k: int = _DEFAULT_K,
    n_sigma: float = _DEFAULT_N_SIGMA,
) -> np.ndarray:
    """统计滤波.

    Args:
        points: (N, 3) float32.
        k: k 近邻数.
        n_sigma: 离群阈值 = mean + n_sigma * std.

    Returns:
        mask: (N,) bool — True 表示保留该点.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    dists = _pairwise_dists(points)
    # 每个 row 取最近 k 个 (含自己) 的平均距离.
    k_actual = min(k, n)
    # argpartition 取前 k 小.
    part_idx = np.argpartition(dists, k_actual - 1, axis=1)[:, :k_actual]
    row_idx = np.arange(n)[:, None]
    knn_dists = dists[row_idx, part_idx]
    avg_dists = knn_dists.mean(axis=1)  # (N,)

    mean_d = float(avg_dists.mean())
    std_d = float(avg_dists.std())
    threshold = mean_d + n_sigma * std_d
    mask = avg_dists <= threshold
    _LOG.debug(
        "statistical filter",
        n_in=int(mask.sum()),
        n_out=int((~mask).sum()),
        threshold=threshold,
    )
    return mask


def radius_outlier_removal(
    points: np.ndarray,
    radius: float = _DEFAULT_RADIUS,
    min_neighbors: int = _DEFAULT_MIN_NEIGHBORS,
) -> np.ndarray:
    """半径滤波.

    Args:
        points: (N, 3) float32.
        radius: 邻域半径 (米).
        min_neighbors: 半径内至少多少邻点才保留.

    Returns:
        mask: (N,) bool.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    dists = _pairwise_dists(points)
    # 半径内邻点数 (含自己).
    counts = (dists <= radius).sum(axis=1)
    mask = counts >= min_neighbors
    _LOG.debug(
        "radius filter",
        n_in=int(mask.sum()),
        n_out=int((~mask).sum()),
        radius=radius,
        min_neighbors=min_neighbors,
    )
    return mask


def filter_statistical_pc(
    pc: PointCloud,
    k: int = _DEFAULT_K,
    n_sigma: float = _DEFAULT_N_SIGMA,
    inplace: bool = True,
) -> PointCloud:
    """给 PointCloud 应用统计滤波."""
    target = (
        pc
        if inplace
        else PointCloud(
            points=pc.points.copy(),
            colors=pc.colors.copy(),
            normals=pc.normals.copy(),
            metadata=pc.metadata.copy(),
        )
    )
    if target.n_points == 0:
        return target
    n_before = target.n_points
    mask = statistical_outlier_removal(target.points, k=k, n_sigma=n_sigma)
    target.points = target.points[mask]
    if target.has_colors:
        target.colors = target.colors[mask]
    if target.has_normals:
        target.normals = target.normals[mask]
    target.metadata["statistical_filter"] = {
        "k": k,
        "n_sigma": n_sigma,
        "n_before": int(n_before),
        "n_after": int(mask.sum()),
    }
    return target


def filter_radius_pc(
    pc: PointCloud,
    radius: float = _DEFAULT_RADIUS,
    min_neighbors: int = _DEFAULT_MIN_NEIGHBORS,
    inplace: bool = True,
) -> PointCloud:
    """给 PointCloud 应用半径滤波."""
    target = (
        pc
        if inplace
        else PointCloud(
            points=pc.points.copy(),
            colors=pc.colors.copy(),
            normals=pc.normals.copy(),
            metadata=pc.metadata.copy(),
        )
    )
    if target.n_points == 0:
        return target
    mask = radius_outlier_removal(target.points, radius=radius, min_neighbors=min_neighbors)
    target.points = target.points[mask]
    if target.has_colors:
        target.colors = target.colors[mask]
    if target.has_normals:
        target.normals = target.normals[mask]
    target.metadata["radius_filter"] = {
        "radius": radius,
        "min_neighbors": min_neighbors,
        "n_before": int(pc.n_points),
        "n_after": int(mask.sum()),
    }
    return target
