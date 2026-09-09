"""M016 点云优化: 法向量估计.

用 PCA 局部拟合估计每个点的法向量:
1. 对每个点找 k 近邻 (brute-force, 纯 numpy, 无 scipy/sklearn).
2. 计算局部协方差矩阵 (3x3).
3. 最小特征值对应的特征向量 = 法向量.
4. 方向统一朝向参考点 (质心或视点), 保证一致朝向.

限制:
- brute-force kNN 是 O(N^2), 适合中小点云 (< 50k 点).
- 大点云需 KDTree (scipy.spatial.cKDTree), 留后续.
"""

from __future__ import annotations

import numpy as np

from modules.logging import get_logger

from .types import PointCloud

_LOG = get_logger("modules.pointcloud.normals")

# 默认 k 近邻数.
_DEFAULT_K = 10
# 默认视点 (用于定向, None=用局部质心).
_DEFAULT_VIEWPOINT = None
# 协方差矩阵维度.
_COV_DIM = 3
# 特征向量索引 (最小特征值).
_MIN_EIG_IDX = 0
# 归一化最小阈值 (避免除零).
_NORM_EPS = 1e-12


def _knn_brute(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """brute-force k 近邻.

    Args:
        points: (N, 3) float32.
        k: 近邻数.

    Returns:
        indices: (N, k) int32 — 每个点的 k 近邻索引 (含自身).
        dists: (N, k) float32 — 对应距离.
    """
    n = points.shape[0]
    k = min(k, n)
    # 距离矩阵 (N, N) — 大点云会爆内存, 仅适合 N < 50k.
    diff = points[:, None, :] - points[None, :, :]  # (N, N, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (N, N)
    # 每个 row 排序, 取前 k 个.
    indices = np.argpartition(dists_sq, k - 1, axis=1)[:, :k]  # (N, k) 近似
    # 精确排序前 k 个.
    row_idx = np.arange(n)[:, None]
    sub_dists = dists_sq[row_idx, indices]  # (N, k)
    sort_order = np.argsort(sub_dists, axis=1)
    indices = indices[row_idx, sort_order]
    dists = np.sqrt(dists_sq[row_idx, indices])
    return indices.astype(np.int32), dists.astype(np.float32)


def estimate_normals(
    points: np.ndarray,
    k: int = _DEFAULT_K,
    viewpoint: np.ndarray | None = _DEFAULT_VIEWPOINT,
) -> np.ndarray:
    """估计每个点的法向量 (PCA 局部拟合).

    Args:
        points: (N, 3) float32/float64.
        k: k 近邻数.
        viewpoint: 视点 (3,). 法向量朝向该点; None=朝向局部质心.

    Returns:
        normals: (N, 3) float32 单位向量.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, _COV_DIM), dtype=np.float32)

    # k 近邻.
    knn_idx, _ = _knn_brute(pts, k)

    normals = np.zeros((n, _COV_DIM), dtype=np.float64)
    for i in range(n):
        neighbors = pts[knn_idx[i]]  # (k, 3)
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid  # (k, 3)
        # 协方差矩阵 (3, 3).
        cov = centered.T @ centered / max(len(neighbors), 1)
        # 特征分解.
        eigvals, eigvecs = np.linalg.eigh(cov)  # 升序.
        normal = eigvecs[:, _MIN_EIG_IDX]  # 最小特征值对应的向量.
        # 归一化.
        norm = np.linalg.norm(normal)
        if norm > _NORM_EPS:
            normal = normal / norm
        # 定向: 朝向视点或局部质心.
        ref = np.asarray(viewpoint, dtype=np.float64) if viewpoint is not None else centroid
        # 法向量应与 (ref - point) 同向.
        to_ref = ref - pts[i]
        if np.dot(normal, to_ref) < 0:
            normal = -normal
        normals[i] = normal

    return normals.astype(np.float32)


def estimate_normals_pc(
    pc: PointCloud,
    k: int = _DEFAULT_K,
    viewpoint: np.ndarray | None = _DEFAULT_VIEWPOINT,
    inplace: bool = True,
) -> PointCloud:
    """给 PointCloud 添加法向量.

    Args:
        pc: 输入点云.
        k: k 近邻数.
        viewpoint: 视点.
        inplace: True 修改 pc; False 返回副本.

    Returns:
        带法向量的 PointCloud.
    """
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
    _LOG.debug("estimating normals", n_points=target.n_points, k=k)
    target.normals = estimate_normals(target.points, k=k, viewpoint=viewpoint)
    target.metadata["normals_estimated"] = True
    target.metadata["normals_k"] = k
    return target
