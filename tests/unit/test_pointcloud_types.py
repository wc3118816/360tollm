"""M015 单元测试: PointCloud 数据结构 + PLY 序列化.

覆盖:
- 构造 + n_points / has_colors / has_normals
- 校验 (形状不匹配/长度不一致)
- has_nan / filter_valid
- merge
- summary
- PLY 保存/加载 (binary + ASCII)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from modules.pointcloud import PointCloud


class TestPointCloudConstruction:
    def test_basic_points_only(self) -> None:
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        pc = PointCloud(points=pts)
        assert pc.n_points == 2
        assert not pc.has_colors
        assert not pc.has_normals

    def test_with_colors(self) -> None:
        pts = np.array([[1, 2, 3]], dtype=np.float32)
        colors = np.array([[255, 0, 0]], dtype=np.uint8)
        pc = PointCloud(points=pts, colors=colors)
        assert pc.has_colors
        assert pc.colors.shape == (1, 3)

    def test_empty_points(self) -> None:
        pc = PointCloud(points=np.zeros((0, 3), dtype=np.float32))
        assert pc.n_points == 0


class TestPointCloudValidation:
    def test_wrong_points_shape(self) -> None:
        bad = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="points must be"):
            PointCloud(points=bad)

    def test_colors_length_mismatch(self) -> None:
        pts = np.zeros((3, 3), dtype=np.float32)
        colors = np.zeros((2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="colors length"):
            PointCloud(points=pts, colors=colors)

    def test_normals_length_mismatch(self) -> None:
        pts = np.zeros((3, 3), dtype=np.float32)
        normals = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="normals length"):
            PointCloud(points=pts, normals=normals)


class TestPointCloudNaN:
    def test_has_nan_true(self) -> None:
        pts = np.array([[1, 2, 3], [float("nan"), 5, 6]], dtype=np.float32)
        pc = PointCloud(points=pts)
        assert pc.has_nan is True

    def test_has_nan_false(self) -> None:
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        pc = PointCloud(points=pts)
        assert pc.has_nan is False

    def test_filter_valid_removes_nan(self) -> None:
        pts = np.array([[1, 2, 3], [float("nan"), 5, 6], [7, 8, 9]], dtype=np.float32)
        colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        pc = PointCloud(points=pts, colors=colors)
        pc.filter_valid()
        assert pc.n_points == 2
        assert not pc.has_nan


class TestPointCloudMerge:
    def test_merge_two_clouds(self) -> None:
        pc1 = PointCloud(points=np.array([[1, 2, 3]], dtype=np.float32))
        pc2 = PointCloud(points=np.array([[4, 5, 6]], dtype=np.float32))
        pc1.merge(pc2)
        assert pc1.n_points == 2

    def test_merge_with_colors(self) -> None:
        pc1 = PointCloud(
            points=np.array([[1, 2, 3]], dtype=np.float32),
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
        )
        pc2 = PointCloud(
            points=np.array([[4, 5, 6]], dtype=np.float32),
            colors=np.array([[0, 255, 0]], dtype=np.uint8),
        )
        pc1.merge(pc2)
        assert pc1.n_points == 2
        assert pc1.has_colors
        assert pc1.colors.shape == (2, 3)


class TestPointCloudSummary:
    def test_summary_fields(self) -> None:
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        pc = PointCloud(points=pts)
        s = pc.summary()
        assert s["n_points"] == 2
        assert s["n_valid"] == 2
        assert s["has_nan"] is False
        assert "bbox_min" in s
        assert "bbox_max" in s
        assert s["bbox_min"] == [1.0, 2.0, 3.0]
        assert s["bbox_max"] == [4.0, 5.0, 6.0]


class TestPointCloudPLY:
    @pytest.fixture
    def tmp_ply(self, tmp_path) -> str:
        return str(tmp_path / "test.ply")

    def test_save_load_binary(self, tmp_ply: str) -> None:
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        pc = PointCloud(points=pts, colors=colors)
        pc.save_ply(tmp_ply, binary=True)
        loaded = PointCloud.load_ply(tmp_ply)
        assert loaded.n_points == 2
        assert np.allclose(loaded.points, pts, atol=1e-4)
        assert np.array_equal(loaded.colors, colors)

    def test_save_load_ascii(self, tmp_ply: str) -> None:
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=pts)
        pc.save_ply(tmp_ply, binary=False)
        loaded = PointCloud.load_ply(tmp_ply)
        assert loaded.n_points == 1
        assert np.allclose(loaded.points, pts, atol=1e-4)
        assert not loaded.has_colors

    def test_save_load_no_colors(self, tmp_ply: str) -> None:
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=pts)
        pc.save_ply(tmp_ply, binary=True)
        loaded = PointCloud.load_ply(tmp_ply)
        assert loaded.n_points == 1
        assert not loaded.has_colors

    def test_file_created(self, tmp_ply: str) -> None:
        pc = PointCloud(points=np.array([[1, 2, 3]], dtype=np.float32))
        pc.save_ply(tmp_ply)
        assert os.path.exists(tmp_ply)
        assert os.path.getsize(tmp_ply) > 0
