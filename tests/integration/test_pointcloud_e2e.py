"""M015 集成测试: 点云生成端到端.

验证:
1. DepthMap + Pose + Image → PointCloud 完整链路
2. AC: 点云无 NaN
3. AC: 在 world 坐标系对齐 (多帧融合后坐标合理)
4. PLY 保存/加载往返一致
5. 多帧融合后点数增加
"""

from __future__ import annotations

import os

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.depth.types import DepthMap as DepthMapType
from modules.pointcloud import PointCloud, PointCloudGenerator
from modules.vo import DummyVisualOdometry
from modules.vo.types import Pose


class TestFullPipeline:
    """Depth + VO + PointCloud 完整链路."""

    def test_frame_to_pointcloud(self) -> None:
        """Frame.image → DepthMap → PointCloud."""
        # 1. 生成图像.
        h, w = 16, 32
        yy, xx = np.mgrid[0:h, 0:w].astype(np.uint8)
        img = np.stack([yy, xx, yy], axis=-1)

        # 2. 估计深度.
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        assert isinstance(dm, DepthMapType)

        # 3. VO 估计位姿.
        vo = DummyVisualOdometry()
        traj = vo.estimate([type("F", (), {"image": img, "frame_id": 0, "timestamp": 0.0})()])
        pose = traj[0]

        # 4. 生成点云.
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, pose)
        assert pc.n_points > 0
        assert not pc.has_nan

    def test_ac_no_nan(self) -> None:
        """AC: 点云无 NaN."""
        est = DummyDepthEstimator()
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        dm = est.estimate(img)
        pose = Pose.identity()
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, pose)
        assert not pc.has_nan
        assert np.isfinite(pc.points).all()

    def test_ac_world_coords_aligned(self) -> None:
        """AC: 在 world 坐标系对齐.

        两帧不同 pose, 生成的点云应在不同 world 位置.
        """
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)

        # pose 1: identity.
        pose1 = Pose.identity(frame_id=0)
        # pose 2: 沿 +x 平移 2m.
        m2 = np.eye(4)
        m2[:3, 3] = [2, 0, 0]
        pose2 = Pose(matrix=m2, frame_id=1)

        gen = PointCloudGenerator()
        pc1 = gen.generate(img, dm, pose1)
        pc2 = gen.generate(img, dm, pose2)

        # 两帧的 world 坐标应不同 (pose2 的点整体偏移).
        centroid1 = pc1.points.mean(axis=0)
        centroid2 = pc2.points.mean(axis=0)
        diff = centroid2 - centroid1
        # 应有显著偏移 (至少一个轴 > 0.5).
        assert np.linalg.norm(diff) > 0.5


class TestMultiFrameFusion:
    """多帧融合."""

    def test_multi_frame_point_count_increases(self) -> None:
        """多帧融合后点数 > 单帧."""
        est = DummyDepthEstimator()
        h, w = 8, 16
        img = np.zeros((h, w, 3), dtype=np.uint8)

        frames = []
        for i in range(3):
            dm = est.estimate(img)
            m = np.eye(4)
            m[:3, 3] = [i * 0.5, 0, 0]
            pose = Pose(matrix=m, frame_id=i, timestamp=float(i))
            f = type("Frame", (), {})()
            f.image = img
            f.depth = dm
            f.pose = pose
            frames.append(f)

        gen = PointCloudGenerator()
        pc_single = gen.generate(frames[0].image, frames[0].depth, frames[0].pose)
        pc_multi = gen.generate_multi(frames)

        assert pc_multi.n_points > pc_single.n_points
        assert pc_multi.metadata["n_frames"] == 3

    def test_multi_frame_metadata(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        frames = []
        for i in range(2):
            dm = est.estimate(img)
            pose = Pose.identity(frame_id=i)
            f = type("Frame", (), {})()
            f.image = img
            f.depth = dm
            f.pose = pose
            frames.append(f)

        gen = PointCloudGenerator()
        pc = gen.generate_multi(frames)
        assert "n_frames" in pc.metadata
        assert "source_frame_ids" in pc.metadata
        assert pc.metadata["n_frames"] == 2


class TestPLERoundTrip:
    """PLY 保存/加载往返一致."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        pose = Pose.identity()
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, pose)

        ply_path = str(tmp_path / "roundtrip.ply")
        pc.save_ply(ply_path, binary=True)
        loaded = PointCloud.load_ply(ply_path)

        assert loaded.n_points == pc.n_points
        assert np.allclose(loaded.points, pc.points, atol=1e-4)
        if pc.has_colors:
            assert np.array_equal(loaded.colors, pc.colors)

    def test_ply_file_exists(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())

        ply_path = str(tmp_path / "exists.ply")
        pc.save_ply(ply_path)
        assert os.path.exists(ply_path)
        assert os.path.getsize(ply_path) > 0


class TestEndToEndWithVO:
    """完整链路: Image → Depth + VO → PointCloud → PLY."""

    def test_full_workflow(self, tmp_path) -> None:
        # 1. 生成 5 帧图像序列.
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

        # 2. 批量深度估计.
        depth_est = DummyDepthEstimator()
        for f in frames:
            f.depth = depth_est.estimate(f.image)

        # 3. VO 估计轨迹.
        vo = DummyVisualOdometry(step_distance=0.5)
        traj = vo.estimate(frames)
        for f, pose in zip(frames, traj, strict=True):
            f.pose = pose

        # 4. 点云融合.
        gen = PointCloudGenerator()
        pc = gen.generate_multi(frames)
        assert pc.n_points > 0
        assert not pc.has_nan
        assert pc.metadata["n_frames"] == 5

        # 5. 保存 PLY.
        ply_path = str(tmp_path / "e2e.ply")
        pc.save_ply(ply_path)
        assert os.path.exists(ply_path)

        # 6. 重新加载.
        loaded = PointCloud.load_ply(ply_path)
        assert loaded.n_points == pc.n_points
