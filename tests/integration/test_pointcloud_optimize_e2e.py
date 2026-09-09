"""M016 集成测试: 点云优化端到端.

验证:
1. 完整流程: 生成 → 优化 (降采样 + 滤波 + 法向量)
2. AC: 法向量已估计
3. AC: 离群点被移除
4. 优化后点数减少
5. 优化后 PLY 保存/加载
"""

from __future__ import annotations

import os

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.pointcloud import (
    OptimizerConfig,
    PointCloud,
    PointCloudGenerator,
    PointCloudOptimizer,
)
from modules.vo import DummyVisualOdometry
from modules.vo.types import Pose


class TestFullOptimizePipeline:
    """生成 → 优化完整链路."""

    def test_generate_then_optimize(self) -> None:
        # 1. 生成点云.
        est = DummyDepthEstimator()
        h, w = 16, 32
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[::2, ::2] = 200  # 添加纹理
        dm = est.estimate(img)
        pose = Pose.identity()
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, pose)
        original_n = pc.n_points

        # 2. 优化.
        opt = PointCloudOptimizer(OptimizerConfig(voxel_size=0.5))
        opt_pc = opt.optimize(pc)
        assert opt_pc.n_points <= original_n

    def test_ac_normals_estimated(self) -> None:
        """AC: 法向量已估计."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        opt = PointCloudOptimizer()
        opt_pc = opt.optimize(pc)
        assert opt_pc.has_normals
        assert opt_pc.normals.shape == (opt_pc.n_points, 3)

    def test_ac_outliers_removed(self) -> None:
        """AC: 离群点被移除."""
        est = DummyDepthEstimator()
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())

        # 添加离群点 (需要重新构造 PointCloud 以保持 colors/points 长度一致).
        outliers = np.random.RandomState(99).rand(5, 3).astype(np.float32) * 50 + 50
        all_points = np.vstack([pc.points, outliers])
        all_colors = np.vstack(
            [
                pc.colors if pc.has_colors else np.zeros((pc.n_points, 3), dtype=np.uint8),
                np.zeros((5, 3), dtype=np.uint8),
            ]
        )
        pc = PointCloud(points=all_points, colors=all_colors)
        n_before = pc.n_points

        opt = PointCloudOptimizer(
            OptimizerConfig(
                voxel_size=0.0,  # 不降采样
                enable_radius_filter=False,  # 只用统计滤波
            )
        )
        opt_pc = opt.optimize(pc)
        # 离群点应被滤波移除.
        assert opt_pc.n_points < n_before


class TestOptimizerSteps:
    def test_voxel_downsample_reduces_points(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        n_before = pc.n_points

        opt = PointCloudOptimizer(
            OptimizerConfig(
                voxel_size=1.0,
                enable_stat_filter=False,
                enable_radius_filter=False,
                enable_normals=False,
            )
        )
        opt_pc = opt.optimize(pc)
        assert opt_pc.n_points < n_before

    def test_filters_reduce_points(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())

        opt = PointCloudOptimizer(
            OptimizerConfig(
                voxel_size=0.0,
                enable_stat_filter=True,
                enable_radius_filter=True,
                enable_normals=False,
            )
        )
        opt_pc = opt.optimize(pc)
        # 滤波后点数 <= 原.
        assert opt_pc.n_points <= pc.n_points

    def test_metadata_updated(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())

        opt = PointCloudOptimizer()
        opt_pc = opt.optimize(pc)
        assert opt_pc.metadata["optimized"] is True
        assert "n_before_optimize" in opt_pc.metadata
        assert "n_after_optimize" in opt_pc.metadata


class TestPLYWithNormals:
    """带法向量的点云 PLY 保存/加载."""

    def test_save_load_with_normals(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())

        opt = PointCloudOptimizer()
        opt_pc = opt.optimize(pc)

        ply_path = str(tmp_path / "with_normals.ply")
        opt_pc.save_ply(ply_path)
        assert os.path.exists(ply_path)

        loaded = PointCloud.load_ply(ply_path)
        assert loaded.n_points == opt_pc.n_points
        # PLY 不保存 normals (当前格式只存 points + colors), 所以不检查.


class TestEndToEndWorkflow:
    """完整工作流: Image → Depth+VO → PointCloud → Optimize → PLY."""

    def test_full_workflow(self, tmp_path) -> None:
        # 1. 5 帧图像.
        h, w = 8, 16
        frames = []
        for i in range(5):
            img = np.full((h, w, 3), 50 + i * 10, dtype=np.uint8)
            f = type("Frame", (), {})()
            f.image = img
            f.frame_id = i
            f.timestamp = float(i)
            f.depth = None
            f.pose = None
            frames.append(f)

        # 2. 深度 + VO.
        depth_est = DummyDepthEstimator()
        for f in frames:
            f.depth = depth_est.estimate(f.image)
        vo = DummyVisualOdometry(step_distance=0.5)
        traj = vo.estimate(frames)
        for f, pose in zip(frames, traj, strict=True):
            f.pose = pose

        # 3. 点云生成 + 优化.
        gen = PointCloudGenerator()
        pc = gen.generate_multi(frames)
        # 用宽松配置 (大半径, 少邻点, 禁滤波避免空点云).
        opt = PointCloudOptimizer(
            OptimizerConfig(
                voxel_size=0.5,
                enable_stat_filter=False,
                enable_radius_filter=False,
                enable_normals=True,
            )
        )
        opt_pc = opt.optimize(pc)
        assert opt_pc.n_points > 0
        assert opt_pc.has_normals

        # 4. PLY 保存.
        ply_path = str(tmp_path / "optimized.ply")
        opt_pc.save_ply(ply_path)
        assert os.path.exists(ply_path)
