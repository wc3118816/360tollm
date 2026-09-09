"""M016 单元测试: 法向量估计.

覆盖:
- 平面点云 → 法向量垂直平面
- 球面点云 → 法向量径向
- 单位长度
- 定向一致 (朝向视点)
- estimate_normals_pc 给 PointCloud 加法向量
- 空点云处理
"""

from __future__ import annotations

import math

import numpy as np

from modules.pointcloud import PointCloud, estimate_normals, estimate_normals_pc


class TestEstimateNormalsPlane:
    """XY 平面上的点 → 法向量 ≈ (0, 0, ±1)."""

    def test_xy_plane_normals(self) -> None:
        # 100 个 XY 平面上的点.
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        xx, yy = np.meshgrid(x, y)
        pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(100)], axis=-1).astype(np.float32)
        normals = estimate_normals(pts, k=10)
        assert normals.shape == (100, 3)
        # 所有法向量应 ≈ (0, 0, ±1).
        z_components = np.abs(normals[:, 2])
        assert (z_components > 0.95).all()

    def test_xz_plane_normals(self) -> None:
        # XZ 平面 → 法向量 ≈ (0, ±1, 0).
        x = np.linspace(-1, 1, 10)
        z = np.linspace(-1, 1, 10)
        xx, zz = np.meshgrid(x, z)
        pts = np.stack([xx.ravel(), np.zeros(100), zz.ravel()], axis=-1).astype(np.float32)
        normals = estimate_normals(pts, k=10)
        y_components = np.abs(normals[:, 1])
        assert (y_components > 0.95).all()


class TestEstimateNormalsSphere:
    """球面点云 → 法向量径向."""

    def test_sphere_normals_radial(self) -> None:
        # 球面上的点 → 法向量应 ≈ 径向方向.
        n = 100
        rng = np.random.RandomState(42)
        # 均匀球面采样.
        phi = rng.uniform(0, 2 * math.pi, n)
        theta = np.arccos(rng.uniform(-1, 1, n))
        pts = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)
        # 视点 = 原点.
        normals = estimate_normals(pts, k=10, viewpoint=np.array([0, 0, 0]))
        # 法向量应与径向方向同向 (指向原点方向, 因为 viewpoint=origin).
        for i in range(n):
            radial = pts[i] / np.linalg.norm(pts[i])
            dot = np.dot(normals[i], radial)
            # 视点在原点, 法向量指向视点 = -radial.
            assert dot < -0.7, f"normal {normals[i]} not anti-parallel to {radial}"


class TestNormalProperties:
    def test_unit_length(self) -> None:
        pts = np.random.RandomState(0).rand(50, 3).astype(np.float32)
        normals = estimate_normals(pts, k=10)
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_points(self) -> None:
        pts = np.zeros((0, 3), dtype=np.float32)
        normals = estimate_normals(pts)
        assert normals.shape == (0, 3)

    def test_single_point(self) -> None:
        pts = np.array([[1, 2, 3]], dtype=np.float32)
        normals = estimate_normals(pts, k=1)
        # 单点 k=1 → 协方差为 0, 法向量未定义, 应返回零或任意.
        assert normals.shape == (1, 3)

    def test_small_k(self) -> None:
        """k 小于点数时正常工作."""
        pts = np.random.RandomState(1).rand(20, 3).astype(np.float32)
        normals = estimate_normals(pts, k=3)
        assert normals.shape == (20, 3)


class TestEstimateNormalsPC:
    def test_adds_normals_to_pointcloud(self) -> None:
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        xx, yy = np.meshgrid(x, y)
        pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(100)], axis=-1).astype(np.float32)
        pc = PointCloud(points=pts)
        assert not pc.has_normals
        estimate_normals_pc(pc, k=10)
        assert pc.has_normals
        assert pc.normals.shape == (100, 3)

    def test_metadata_updated(self) -> None:
        pc = PointCloud(points=np.random.rand(20, 3).astype(np.float32))
        estimate_normals_pc(pc, k=5)
        assert pc.metadata["normals_estimated"] is True
        assert pc.metadata["normals_k"] == 5

    def test_not_inplace(self) -> None:
        pts = np.random.RandomState(0).rand(20, 3).astype(np.float32)
        pc = PointCloud(points=pts)
        pc2 = estimate_normals_pc(pc, k=5, inplace=False)
        assert not pc.has_normals
        assert pc2.has_normals
