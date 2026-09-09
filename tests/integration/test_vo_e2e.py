"""M011 集成测试: Visual Odometry 端到端.

验证:
1. 从 Frame 序列到 Trajectory 完整链路
2. AC: 在回放数据上输出连续轨迹
3. AC: 轨迹无 NaN
4. AC: 失败率可统计
5. 多帧后轨迹位移合理
6. Pose → Frame.pose 注入
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.vo import DummyVisualOdometry, Trajectory, create_vo
from modules.vo.base import VisualOdometry


class _Frame:
    """最小 Frame 替身 (含 .image / .timestamp / .frame_id / .pose)."""

    def __init__(self, frame_id: int, timestamp: float | None = None) -> None:
        self.frame_id = frame_id
        self.timestamp = timestamp if timestamp is not None else float(frame_id)
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.pose = None
        self.source = "test"


class TestFrameToTrajectory:
    """Frame 序列 → Trajectory."""

    def test_estimate_from_frames(self) -> None:
        vo = DummyVisualOdometry()
        frames = [_Frame(i) for i in range(5)]
        traj = vo.estimate(frames)
        assert isinstance(traj, Trajectory)
        assert len(traj) == 5

    def test_inject_pose_into_frame(self) -> None:
        """Trajectory.poses 可注入回 Frame.pose."""
        vo = DummyVisualOdometry()
        frames = [_Frame(i) for i in range(3)]
        traj = vo.estimate(frames)
        for frame, pose in zip(frames, traj, strict=True):
            frame.pose = pose.matrix
        assert frames[0].pose is not None
        assert frames[0].pose.shape == (4, 4)


class TestACCompliance:
    """M011 AC 三项要求."""

    def test_continuous_trajectory(self) -> None:
        """AC: 能在回放数据上输出连续轨迹."""
        vo = DummyVisualOdometry(step_distance=0.1)
        frames = [_Frame(i, timestamp=float(i) / 10) for i in range(20)]
        traj = vo.estimate(frames)
        # 连续性: 时间戳单调递增.
        ts = traj.timestamps
        assert np.all(np.diff(ts) > 0)
        # 长度 == 输入帧数.
        assert len(traj) == 20

    def test_trajectory_no_nan(self) -> None:
        """AC: 轨迹无 NaN."""
        vo = DummyVisualOdometry()
        frames = [_Frame(i) for i in range(50)]
        traj = vo.estimate(frames)
        assert traj.has_nan() is False
        # 显式检查每个 pose 矩阵.
        for p in traj:
            assert np.isfinite(p.matrix).all()

    def test_failure_rate_stats(self) -> None:
        """AC: 失败率可统计."""
        vo = DummyVisualOdometry()
        frames = [_Frame(i) for i in range(10)]
        vo.estimate(frames)
        stats = vo.stats
        assert stats.n_frames == 10
        assert stats.failure_rate == 0.0  # Dummy 不失败
        d = stats.as_dict()
        assert "failure_rate" in d
        assert "n_success" in d
        assert "n_fail" in d


class TestTrajectoryMetrics:
    def test_displacement_increases_with_frames(self) -> None:
        """帧数越多, 位移越大."""
        vo1 = DummyVisualOdometry(step_distance=0.5)
        vo2 = DummyVisualOdometry(step_distance=0.5)
        traj_short = vo1.estimate([_Frame(i) for i in range(5)])
        traj_long = vo2.estimate([_Frame(i) for i in range(20)])
        assert traj_long.total_displacement() > traj_short.total_displacement()

    def test_linear_motion_displacement(self) -> None:
        """沿 +x 匀速直线: displacement = (n-1) * step."""
        vo = DummyVisualOdometry(step_distance=0.5)
        traj = vo.estimate([_Frame(i) for i in range(11)])
        # 10 步 * 0.5 = 5.0 米
        assert traj.total_displacement() == pytest.approx(5.0, abs=1e-9)

    def test_path_length_geq_displacement(self) -> None:
        """路径长度 >= 直线位移."""
        vo = DummyVisualOdometry(step_distance=0.3, yaw_amplitude=0.5)
        traj = vo.estimate([_Frame(i) for i in range(20)])
        assert traj.total_path_length() >= traj.total_displacement() - 1e-9

    def test_summary_contains_required_fields(self) -> None:
        vo = DummyVisualOdometry()
        traj = vo.estimate([_Frame(i) for i in range(5)])
        s = traj.summary()
        assert "n_poses" in s
        assert "has_nan" in s
        assert "total_displacement" in s
        assert "total_path_length" in s
        assert "backend" in s


class TestFromSettings:
    def test_create_from_default_config(self) -> None:
        vo = create_vo()
        assert isinstance(vo, VisualOdometry)
        traj = vo.estimate([_Frame(i) for i in range(5)])
        assert len(traj) == 5
        assert traj.has_nan() is False


class TestEdgeCases:
    def test_single_frame(self) -> None:
        """单帧 → 单 pose (identity)."""
        vo = DummyVisualOdometry()
        traj = vo.estimate([_Frame(0)])
        assert len(traj) == 1
        assert traj.total_displacement() == 0.0

    def test_empty_input(self) -> None:
        vo = DummyVisualOdometry()
        traj = vo.estimate([])
        assert len(traj) == 0
