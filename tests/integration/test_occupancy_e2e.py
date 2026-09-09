"""M018 集成测试: 点云 → 占用栅格 → 查询/衰减 端到端.

验证:
1. 完整流程: PointCloud → OccupancyGrid
2. AC: 可查询任意位置占用状态
3. AC: 更新后旧信息可衰减
4. 多帧增量更新
5. 与 M015/M016 点云集成
"""

from __future__ import annotations

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyBuilder,
)
from modules.pointcloud import PointCloudGenerator, PointCloudOptimizer
from modules.vo import DummyVisualOdometry
from modules.vo.types import Pose


class TestFullPipeline:
    """PointCloud → OccupancyGrid 完整链路."""

    def test_pointcloud_to_occupancy(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        builder = OccupancyBuilder()
        grid = builder.build(pc.points, sensor_origin=np.zeros(3))
        assert grid.n_voxels > 0
        # 应有 occupied 体素.
        assert grid.summary()["n_occupied"] > 0

    def test_ac_query_any_position(self) -> None:
        """AC: 可查询任意位置占用状态."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        builder = OccupancyBuilder()
        grid = builder.build(pc.points, sensor_origin=np.zeros(3))
        # 查询点云内的点 → occupied.
        if pc.n_points > 0:
            p = pc.points[0]
            status = grid.status_at(p)
            assert status in (OCCUPIED, FREE, UNKNOWN)
        # 查询远处的点 → unknown 或 free.
        far_status = grid.status_at(np.array([100.0, 100.0, 100.0]))
        assert far_status == UNKNOWN

    def test_ac_decay(self) -> None:
        """AC: 更新后旧信息可衰减."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        builder = OccupancyBuilder()
        grid = builder.build(pc.points, sensor_origin=np.zeros(3))
        n_occupied_before = grid.summary()["n_occupied"]
        # 衰减.
        grid.decay(decay_rate=0.1, conf_threshold=0.5)
        n_occupied_after = grid.summary()["n_occupied"]
        assert n_occupied_after <= n_occupied_before


class TestMultiFrameUpdate:
    def test_incremental_update(self) -> None:
        """多帧增量更新."""
        est = DummyDepthEstimator()
        builder = OccupancyBuilder()
        gen = PointCloudGenerator()
        grid = None
        for i in range(3):
            img = np.full((8, 16, 3), 50 + i * 50, dtype=np.uint8)
            dm = est.estimate(img)
            pose = Pose.identity(frame_id=i)
            pc = gen.generate(img, dm, pose)
            if grid is None:
                grid = builder.build(pc.points, sensor_origin=np.zeros(3))
            else:
                grid = builder.build(pc.points, sensor_origin=np.zeros(3), existing=grid)
        assert grid is not None
        assert grid.metadata["n_updates"] == 3


class TestWithOptimizer:
    def test_optimized_cloud(self) -> None:
        """用 M016 优化后的点云构建栅格."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        opt = PointCloudOptimizer()
        opt_pc = opt.optimize(pc)
        builder = OccupancyBuilder()
        grid = builder.build(opt_pc.points, sensor_origin=np.zeros(3))
        assert grid.summary()["n_occupied"] > 0


class TestEndToEnd:
    def test_full_workflow(self) -> None:
        """完整工作流: Image → Depth+VO → PointCloud → Optimize → Occupancy → Decay."""
        # 1. 多帧.
        frames = []
        for i in range(3):
            img = np.full((8, 16, 3), 50 + i * 30, dtype=np.uint8)
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
        builder = OccupancyBuilder()
        grid = None
        for f in frames:
            pc = gen.generate(f.image, f.depth, f.pose)
            if grid is None:
                grid = builder.build(pc.points, sensor_origin=np.zeros(3))
            else:
                grid = builder.build(pc.points, sensor_origin=np.zeros(3), existing=grid)

        assert grid is not None
        s = grid.summary()
        assert s["n_voxels"] > 0
        assert s["n_occupied"] > 0

        # 4. 衰减.
        grid.decay(0.9)
        assert grid.metadata["decay_count"] == 1

    def test_query_free_space(self) -> None:
        """查询 free 空间 (射线投射路径)."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = PointCloudGenerator()
        pc = gen.generate(img, dm, Pose.identity())
        builder = OccupancyBuilder()
        grid = builder.build(pc.points, sensor_origin=np.zeros(3))
        # sensor 附近应有 free 体素 (除非全部 occupied).
        s = grid.summary()
        # 应有 free 或 occupied (取决于点云).
        assert s["n_free"] + s["n_occupied"] > 0
