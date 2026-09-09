"""M016 单元测试: 离群点滤波.

覆盖:
- 统计滤波: 移除远离群点的孤立点
- 半径滤波: 移除半径内邻点不足的点
- filter_*_pc 给 PointCloud 应用滤波
- 空点云处理
- metadata 记录
"""

from __future__ import annotations

import numpy as np

from modules.pointcloud import (
    PointCloud,
    filter_radius_pc,
    filter_statistical_pc,
    radius_outlier_removal,
    statistical_outlier_removal,
)


def _make_dense_cloud(n: int = 100, seed: int = 42) -> np.ndarray:
    """生成密集点云."""
    return np.random.RandomState(seed).rand(n, 3).astype(np.float32) * 0.5


def _add_outliers(pts: np.ndarray, n_outliers: int = 5, far: float = 10.0) -> np.ndarray:
    """在远离主群的位置添加离群点."""
    outliers = np.random.RandomState(99).rand(n_outliers, 3).astype(np.float32) * far + far
    return np.vstack([pts, outliers])


class TestStatisticalOutlierRemoval:
    def test_removes_outliers(self) -> None:
        pts = _make_dense_cloud(100)
        pts_with = _add_outliers(pts, 5)
        mask = statistical_outlier_removal(pts_with, k=10, n_sigma=2.0)
        # 应保留大部分主群点, 移除离群.
        assert mask.sum() >= 90  # 至少 90 个主群点
        assert mask.sum() < 105  # 不应保留全部 (有离群被移除)

    def test_no_outliers_keeps_all(self) -> None:
        pts = _make_dense_cloud(50)
        mask = statistical_outlier_removal(pts, k=5, n_sigma=3.0)
        # 无离群 → 全部保留.
        assert mask.sum() == 50

    def test_empty_points(self) -> None:
        mask = statistical_outlier_removal(np.zeros((0, 3), dtype=np.float32))
        assert mask.shape == (0,)

    def test_strict_threshold_removes_more(self) -> None:
        """n_sigma 小 → 移除更多点."""
        pts = _make_dense_cloud(100)
        mask_loose = statistical_outlier_removal(pts, k=10, n_sigma=3.0)
        mask_strict = statistical_outlier_removal(pts, k=10, n_sigma=0.5)
        assert mask_strict.sum() <= mask_loose.sum()


class TestRadiusOutlierRemoval:
    def test_removes_isolated(self) -> None:
        pts = _make_dense_cloud(100)
        # 添加 5 个远离的点.
        pts_with = _add_outliers(pts, 5, far=10.0)
        mask = radius_outlier_removal(pts_with, radius=0.5, min_neighbors=5)
        # 主群点保留, 离群移除.
        assert mask.sum() >= 90
        assert mask.sum() <= 100

    def test_empty_points(self) -> None:
        mask = radius_outlier_removal(np.zeros((0, 3), dtype=np.float32))
        assert mask.shape == (0,)

    def test_large_radius_keeps_all(self) -> None:
        pts = _make_dense_cloud(20)
        mask = radius_outlier_removal(pts, radius=100.0, min_neighbors=1)
        assert mask.sum() == 20

    def test_large_min_neighbors_removes_all(self) -> None:
        pts = _make_dense_cloud(10)
        # 邻点阈值 > 点数 → 全部移除.
        mask = radius_outlier_removal(pts, radius=0.1, min_neighbors=20)
        assert mask.sum() == 0


class TestFilterPointCloud:
    def test_filter_statistical_pc(self) -> None:
        # 密集云 + 离群点 → 移除离群.
        pts = _make_dense_cloud(50)
        pts_with = _add_outliers(pts, 3)
        pc = PointCloud(points=pts_with)
        n_before = pc.n_points
        filtered = filter_statistical_pc(pc, k=5, n_sigma=2.0)
        assert filtered.n_points < n_before  # 移除了离群
        assert filtered.n_points > 40  # 主群保留

    def test_filter_radius_pc(self) -> None:
        pts = _make_dense_cloud(50)
        pts_with = _add_outliers(pts, 3)
        pc = PointCloud(points=pts_with)
        n_before = pc.n_points
        filtered = filter_radius_pc(pc, radius=0.5, min_neighbors=3)
        assert filtered.n_points < n_before

    def test_filter_preserves_colors(self) -> None:
        pts = _make_dense_cloud(50)
        pts_with = _add_outliers(pts, 3)
        colors = np.random.RandomState(0).randint(0, 255, (53, 3)).astype(np.uint8)
        pc = PointCloud(points=pts_with, colors=colors)
        filtered = filter_statistical_pc(pc, k=5, n_sigma=2.0)
        assert filtered.has_colors
        assert filtered.colors.shape[1] == 3
        # 颜色长度与过滤后的 points 一致.
        assert filtered.colors.shape[0] == filtered.n_points

    def test_metadata_recorded(self) -> None:
        pts = _make_dense_cloud(50)
        pts_with = _add_outliers(pts, 3)
        pc = PointCloud(points=pts_with)
        filtered = filter_statistical_pc(pc, k=5, n_sigma=2.0)
        assert "statistical_filter" in filtered.metadata
        assert filtered.metadata["statistical_filter"]["n_before"] == 53
        assert "n_after" in filtered.metadata["statistical_filter"]

    def test_not_inplace(self) -> None:
        pts = _make_dense_cloud(50)
        pc = PointCloud(points=pts)
        filtered = filter_statistical_pc(pc, k=5, n_sigma=2.0, inplace=False)
        assert pc.n_points == 50  # 原始未变
        assert filtered.n_points <= 50

    def test_empty_pc(self) -> None:
        pc = PointCloud(points=np.zeros((0, 3), dtype=np.float32))
        filtered = filter_statistical_pc(pc)
        assert filtered.n_points == 0
