"""M015 单元测试: PointCloudGenerator + voxel_downsample.

覆盖:
- equirect 模式: 生成点云形状/世界坐标/无 NaN
- perspective 模式: 内参 K 反投影
- 无效深度过滤
- 多帧融合
- 体素降采样
- 颜色提取
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.depth.types import DepthMap
from modules.pointcloud import (
    PointCloud,
    PointCloudGenerator,
    PointCloudGeneratorConfig,
    voxel_downsample,
)
from modules.vo.types import Pose


def _make_depth_map(
    h: int = 16,
    w: int = 32,
    depth_value: float = 5.0,
) -> DepthMap:
    """生成恒定深度的 DepthMap."""
    depth = np.full((h, w), depth_value, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    return DepthMap(depth=depth, confidence=conf, backend="dummy")


def _make_image(h: int = 16, w: int = 32) -> np.ndarray:
    """生成 RGB 测试图像."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 128  # 红色通道
    return img


class TestEquirectGeneration:
    def test_generate_returns_pointcloud(self) -> None:
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect"))
        dm = _make_depth_map(8, 16)
        pose = Pose.identity()
        pc = gen.generate(_make_image(8, 16), dm, pose)
        assert isinstance(pc, PointCloud)
        assert pc.n_points > 0

    def test_point_count_matches_valid_pixels(self) -> None:
        """全有效深度 → 点数 = (H/step) * (W/step)."""
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect", step=1))
        dm = _make_depth_map(8, 16)
        pose = Pose.identity()
        pc = gen.generate(_make_image(8, 16), dm, pose)
        # 8 * 16 = 128 像素, 全部有效.
        assert pc.n_points == 128

    def test_no_nan(self) -> None:
        """AC: 点云无 NaN."""
        gen = PointCloudGenerator()
        dm = _make_depth_map(8, 16)
        pose = Pose.identity()
        pc = gen.generate(_make_image(8, 16), dm, pose)
        assert not pc.has_nan

    def test_world_coords_with_identity_pose(self) -> None:
        """identity pose → world = camera 坐标."""
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect"))
        dm = _make_depth_map(2, 4, depth_value=5.0)
        pose = Pose.identity()
        pc = gen.generate(None, dm, pose)
        # equirect: 中心像素 (u=2, v=1) → ray ≈ (1, 0, 0), point = 5 * (1,0,0) = (5, 0, 0)
        # 检查所有点在深度=5 的球面上 (范数 ≈ 5).
        norms = np.linalg.norm(pc.points, axis=1)
        assert np.allclose(norms, 5.0, atol=0.5)

    def test_world_coords_with_translation(self) -> None:
        """pose 平移 → world 坐标偏移."""
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect"))
        dm = _make_depth_map(4, 8, depth_value=5.0)
        # pose: world 原点在 camera 前方 2m (t=[2,0,0]).
        m = np.eye(4)
        m[:3, 3] = [2, 0, 0]
        pose = Pose(matrix=m)
        pc = gen.generate(None, dm, pose)
        # identity pose 时点范数=5, 平移 pose 后世界坐标 = camera - t 的反变换.
        # 大致检查形状.
        assert pc.n_points > 0
        assert not pc.has_nan


class TestPerspectiveGeneration:
    def test_generate_with_K(self) -> None:
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="perspective"))
        K = np.array([[320, 0, 160], [0, 320, 120], [0, 0, 1]], dtype=np.float64)
        dm = _make_depth_map(8, 16, depth_value=2.0)
        pose = Pose.identity()
        pc = gen.generate(None, dm, pose, camera_matrix=K)
        assert pc.n_points > 0
        assert not pc.has_nan

    def test_perspective_requires_K(self) -> None:
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="perspective"))
        dm = _make_depth_map(4, 4)
        pose = Pose.identity()
        with pytest.raises(ValueError, match="requires camera_matrix"):
            gen.generate(None, dm, pose)

    def test_perspective_center_pixel(self) -> None:
        """中心像素 (主点) → ray = (0, 0, 1), point = depth * (0,0,1)."""
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="perspective", step=1))
        # 内参主点 = (0, 0) (图像中心像素).
        K = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=np.float64)
        # 1x1 深度图, 像素 (0, 0) = 主点.
        dm = DepthMap(
            depth=np.array([[2.0]], dtype=np.float32),
            confidence=np.array([[1.0]], dtype=np.float32),
        )
        pose = Pose.identity()
        pc = gen.generate(None, dm, pose, camera_matrix=K)
        # ray = K^-1 @ [0, 0, 1] = (0, 0, 1) (主点在原点).
        # point = 2 * (0, 0, 1) = (0, 0, 2).
        assert np.allclose(pc.points[0], [0, 0, 2], atol=1e-3)


class TestDepthFiltering:
    def test_zero_depth_filtered(self) -> None:
        """depth=0 的像素被过滤."""
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect"))
        depth = np.full((4, 8), 5.0, dtype=np.float32)
        depth[0, 0] = 0.0  # 无效
        dm = DepthMap(depth=depth, confidence=np.ones_like(depth))
        pose = Pose.identity()
        pc = gen.generate(None, dm, pose)
        # 32 像素 - 1 无效 = 31
        assert pc.n_points == 31

    def test_max_depth_filtered(self) -> None:
        """超过 max_depth 的像素被过滤."""
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect", max_depth=10.0))
        depth = np.full((4, 4), 5.0, dtype=np.float32)
        depth[0, 0] = 50.0  # 太远
        dm = DepthMap(depth=depth, confidence=np.ones_like(depth))
        pose = Pose.identity()
        pc = gen.generate(None, dm, pose)
        assert pc.n_points == 15  # 16 - 1

    def test_step_downsamples(self) -> None:
        """step=2 时点数减半."""
        gen1 = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect", step=1))
        gen2 = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect", step=2))
        dm = _make_depth_map(8, 8)
        pose = Pose.identity()
        pc1 = gen1.generate(None, dm, pose)
        pc2 = gen2.generate(None, dm, pose)
        # step=2: 4x4 = 16 点 (vs 64)
        assert pc2.n_points < pc1.n_points
        assert pc2.n_points == 16


class TestColorExtraction:
    def test_colors_extracted(self) -> None:
        gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect"))
        dm = _make_depth_map(4, 8)
        img = _make_image(4, 8)
        pose = Pose.identity()
        pc = gen.generate(img, dm, pose)
        assert pc.has_colors
        assert pc.colors.shape == (pc.n_points, 3)
        assert (pc.colors[:, 0] == 128).all()  # 红色通道

    def test_no_image_no_colors(self) -> None:
        gen = PointCloudGenerator()
        dm = _make_depth_map(4, 4)
        pose = Pose.identity()
        pc = gen.generate(None, dm, pose)
        assert not pc.has_colors


class TestMultiFrameFusion:
    def _make_frame(self, frame_id: int, depth_value: float = 5.0) -> tuple:
        """生成 (image, depth_map, pose) 三元组."""
        img = _make_image(4, 8)
        dm = _make_depth_map(4, 8, depth_value=depth_value)
        # 沿 +x 平移的 pose.
        m = np.eye(4)
        m[:3, 3] = [frame_id * 0.5, 0, 0]
        pose = Pose(matrix=m, frame_id=frame_id)
        return img, dm, pose

    def test_multi_frame_merge(self) -> None:
        gen = PointCloudGenerator()
        frames = []
        for fid in range(3):
            img, dm, pose = self._make_frame(fid)
            frame = type("Frame", (), {})()
            frame.image = img
            frame.depth = dm
            frame.pose = pose
            frames.append(frame)
        pc = gen.generate_multi(frames)
        assert pc.n_points > 0
        assert "n_frames" in pc.metadata
        assert pc.metadata["n_frames"] == 3

    def test_multi_frame_empty(self) -> None:
        gen = PointCloudGenerator()
        pc = gen.generate_multi([])
        assert pc.n_points == 0


class TestVoxelDownsample:
    def test_downsample_reduces_points(self) -> None:
        # 100 个密集点 (同一体素内).
        pts = np.random.RandomState(42).rand(100, 3).astype(np.float32) * 0.1
        down, _ = voxel_downsample(pts, voxel_size=1.0)
        # 所有点在同一体素 → 1 个点.
        assert down.shape[0] == 1

    def test_downsample_preserves_extent(self) -> None:
        # 2 个远点.
        pts = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
        down, _ = voxel_downsample(pts, voxel_size=1.0)
        assert down.shape[0] == 2

    def test_downsample_with_colors(self) -> None:
        pts = np.array([[0, 0, 0], [0.1, 0.1, 0.1]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        down, down_colors = voxel_downsample(pts, voxel_size=1.0, colors=colors)
        assert down.shape[0] == 1
        assert down_colors.shape == (1, 3)

    def test_empty_input(self) -> None:
        pts = np.zeros((0, 3), dtype=np.float32)
        down, down_colors = voxel_downsample(pts, voxel_size=0.5)
        assert down.shape[0] == 0
        assert down_colors.shape == (0, 3)
