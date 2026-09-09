"""M011 单元测试: VisualOdometry + DummyVO + VOStats + 工厂.

覆盖:
- DummyVO: 轨迹长度/连续性/无 NaN/确定性/位移
- VOStats: 成功/失败/失败率/连续失败/avg_latency
- 工厂: 从 cfg 创建 + 未知 backend 报错 + FeatureVO 无 cv2 报错
- Trajectory: 首帧为单位 pose
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.config.settings import SLAMCfg
from modules.vo import (
    DummyVisualOdometry,
    Trajectory,
    VisualOdometry,
    VOStats,
    create_vo,
)


class _DummyFrame:
    """最小 Frame 替身 (有 .image / .timestamp / .frame_id)."""

    def __init__(self, frame_id: int, timestamp: float | None = None) -> None:
        self.frame_id = frame_id
        self.timestamp = timestamp if timestamp is not None else float(frame_id)
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)


class TestDummyVisualOdometry:
    def test_estimate_returns_trajectory(self) -> None:
        vo = DummyVisualOdometry(step_distance=0.5)
        frames = [_DummyFrame(i) for i in range(10)]
        traj = vo.estimate(frames)
        assert isinstance(traj, Trajectory)
        assert traj.backend == "dummy"

    def test_trajectory_length_matches_frames(self) -> None:
        vo = DummyVisualOdometry()
        frames = [_DummyFrame(i) for i in range(5)]
        traj = vo.estimate(frames)
        assert len(traj) == 5

    def test_first_pose_is_identity(self) -> None:
        """首帧应在 world 原点."""
        vo = DummyVisualOdometry()
        frames = [_DummyFrame(i) for i in range(3)]
        traj = vo.estimate(frames)
        first = traj[0]
        assert np.allclose(first.position, [0, 0, 0], atol=1e-9)

    def test_trajectory_no_nan(self) -> None:
        """AC: 轨迹无 NaN."""
        vo = DummyVisualOdometry()
        frames = [_DummyFrame(i) for i in range(20)]
        traj = vo.estimate(frames)
        assert traj.has_nan() is False

    def test_deterministic_output(self) -> None:
        """相同输入 → 相同输出."""
        vo1 = DummyVisualOdometry(step_distance=0.3)
        vo2 = DummyVisualOdometry(step_distance=0.3)
        frames = [_DummyFrame(i) for i in range(5)]
        t1 = vo1.estimate(frames)
        t2 = vo2.estimate(frames)
        for p1, p2 in zip(t1, t2, strict=True):
            assert np.array_equal(p1.matrix, p2.matrix)

    def test_displacement(self) -> None:
        """沿 +x 移动, displacement = (n-1) * step."""
        vo = DummyVisualOdometry(step_distance=0.5)
        frames = [_DummyFrame(i) for i in range(5)]
        traj = vo.estimate(frames)
        # 4 步 * 0.5 = 2.0 米
        assert traj.total_displacement() == pytest.approx(2.0, abs=1e-9)

    def test_yaw_oscillation(self) -> None:
        """yaw_amplitude > 0 时, pose 有非零 yaw."""
        vo = DummyVisualOdometry(yaw_amplitude=0.5)
        frames = [_DummyFrame(i) for i in range(10)]
        traj = vo.estimate(frames)
        # 至少有一个 pose 的 |yaw| > 0
        yaws = [p.yaw for p in traj]
        assert max(abs(y) for y in yaws) > 0.01

    def test_empty_frames(self) -> None:
        vo = DummyVisualOdometry()
        traj = vo.estimate([])
        assert len(traj) == 0

    def test_describe(self) -> None:
        vo = DummyVisualOdometry(step_distance=0.2)
        d = vo.describe()
        assert d["backend"] == "dummy"
        assert d["step_distance"] == 0.2
        assert d["synthetic"] is True

    def test_stats_accumulated(self) -> None:
        vo = DummyVisualOdometry()
        frames = [_DummyFrame(i) for i in range(10)]
        vo.estimate(frames)
        assert vo.stats.n_frames == 10
        assert vo.stats.n_success == 10
        assert vo.stats.failure_rate == 0.0


class TestVOStats:
    def test_empty(self) -> None:
        s = VOStats()
        assert s.n_frames == 0
        assert s.failure_rate == 0.0
        assert s.avg_latency_ms == 0.0

    def test_success_fail_mix(self) -> None:
        s = VOStats()
        for _ in range(7):
            s.record_success(10.0)
        for _ in range(3):
            s.record_fail(5.0)
        assert s.n_frames == 10
        assert s.n_success == 7
        assert s.n_fail == 3
        assert s.failure_rate == pytest.approx(0.3)
        assert s.success_rate == pytest.approx(0.7)

    def test_consecutive_fails(self) -> None:
        s = VOStats()
        for _ in range(10):
            s.record_fail()
        assert s.consecutive_fails == 10
        assert s.is_failed()

    def test_success_resets_consecutive(self) -> None:
        s = VOStats()
        for _ in range(5):
            s.record_fail()
        s.record_success(1.0)
        assert s.consecutive_fails == 0

    def test_as_dict(self) -> None:
        s = VOStats(backend="dummy")
        s.record_success(5.0)
        d = s.as_dict()
        assert d["backend"] == "dummy"
        assert d["n_frames"] == 1
        assert d["failure_rate"] == 0.0


class TestFactory:
    def test_create_dummy_from_cfg(self) -> None:
        cfg = SLAMCfg(backend="dummy")
        vo = create_vo(cfg)
        assert isinstance(vo, DummyVisualOdometry)

    def test_create_dummy_from_none_backend(self) -> None:
        cfg = SLAMCfg(backend="none")
        vo = create_vo(cfg)
        assert isinstance(vo, DummyVisualOdometry)

    def test_unknown_backend_raises(self) -> None:
        cfg = SLAMCfg(backend="nonexistent")
        with pytest.raises(ValueError, match="unknown VO backend"):
            create_vo(cfg)

    def test_create_from_settings(self) -> None:
        vo = create_vo()
        assert isinstance(vo, VisualOdometry)

    def test_feature_vo_without_cv2_raises(self) -> None:
        from modules.vo.feature_vo import _CV2_AVAILABLE

        if not _CV2_AVAILABLE:
            cfg = SLAMCfg(backend="orb")
            with pytest.raises(RuntimeError, match="requires opencv"):
                create_vo(cfg)
