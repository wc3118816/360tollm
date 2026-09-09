"""M011 单元测试: Pose + Trajectory 数据结构.

覆盖:
- Pose 构造 + R/t/position 属性 + identity
- Pose 校验 (形状/NaN/confidence)
- Pose to_dict
- Trajectory 构造 + positions/timestamps
- Trajectory has_nan / total_displacement / total_path_length / summary
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from modules.vo import Pose, Trajectory


class TestPoseConstruction:
    def test_identity(self) -> None:
        p = Pose.identity(timestamp=1.0, frame_id=5)
        assert np.allclose(p.matrix, np.eye(4))
        assert p.timestamp == 1.0
        assert p.frame_id == 5
        assert p.confidence == 1.0

    def test_basic_construction(self) -> None:
        m = np.eye(4)
        p = Pose(matrix=m, timestamp=2.0, frame_id=10, confidence=0.8)
        assert p.timestamp == 2.0
        assert p.frame_id == 10
        assert p.confidence == 0.8


class TestPoseValidation:
    def test_wrong_shape_rejected(self) -> None:
        bad = np.zeros((3, 3))
        with pytest.raises(ValueError, match="matrix must be"):
            Pose(matrix=bad)

    def test_nan_rejected(self) -> None:
        m = np.eye(4)
        m[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            Pose(matrix=m)

    def test_inf_rejected(self) -> None:
        m = np.eye(4)
        m[0, 0] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            Pose(matrix=m)

    def test_confidence_out_of_range_rejected(self) -> None:
        m = np.eye(4)
        with pytest.raises(ValueError, match="confidence must be in"):
            Pose(matrix=m, confidence=1.5)
        with pytest.raises(ValueError, match="confidence must be in"):
            Pose(matrix=m, confidence=-0.1)


class TestPoseProperties:
    def test_R_t_extraction(self) -> None:
        m = np.eye(4)
        m[:3, 3] = [1, 2, 3]
        p = Pose(matrix=m)
        assert np.allclose(p.R, np.eye(3))
        assert np.allclose(p.t, [1, 2, 3])

    def test_position_identity(self) -> None:
        """identity pose → position = (0,0,0)."""
        p = Pose.identity()
        assert np.allclose(p.position, [0, 0, 0])

    def test_position_translated(self) -> None:
        """t=(1,0,0) 时, position=(-1,0,0) (world 原点在 camera 前方 1m 处)."""
        m = np.eye(4)
        m[:3, 3] = [1, 0, 0]
        p = Pose(matrix=m)
        assert np.allclose(p.position, [-1, 0, 0])

    def test_yaw_extraction(self) -> None:
        """yaw=π/2 的旋转 → yaw 属性 ≈ π/2."""
        from modules.projection.rotation import rotation_yaw

        R = rotation_yaw(math.pi / 2)
        m = np.eye(4)
        m[:3, :3] = R
        p = Pose(matrix=m)
        assert p.yaw == pytest.approx(math.pi / 2, abs=1e-9)


class TestPoseSerialization:
    def test_to_dict(self) -> None:
        p = Pose.identity(timestamp=1.5, frame_id=3)
        p.confidence = 0.9
        d = p.to_dict()
        assert d["timestamp"] == 1.5
        assert d["frame_id"] == 3
        assert d["confidence"] == 0.9
        assert d["position"] == [0.0, 0.0, 0.0]
        assert d["yaw"] == 0.0
        assert "matrix" in d


class TestTrajectory:
    def _make_trajectory(self, n: int = 5) -> Trajectory:
        poses = [
            Pose(
                matrix=np.eye(4) * 1,
                timestamp=float(i),
                frame_id=i,
            )
            for i in range(n)
        ]
        # 修正矩阵 (eye(4)*1 会把对角线变成 1, 但其它位置 0).
        for p in poses:
            p.matrix = np.eye(4)
        return Trajectory(poses=poses, backend="dummy")

    def test_construction(self) -> None:
        traj = self._make_trajectory(5)
        assert len(traj) == 5
        assert traj[0].frame_id == 0
        assert traj[-1].frame_id == 4

    def test_positions(self) -> None:
        traj = self._make_trajectory(3)
        positions = traj.positions
        assert positions.shape == (3, 3)

    def test_timestamps(self) -> None:
        traj = self._make_trajectory(3)
        ts = traj.timestamps
        assert ts.shape == (3,)
        assert list(ts) == [0.0, 1.0, 2.0]

    def test_empty_trajectory(self) -> None:
        traj = Trajectory(poses=[], backend="dummy")
        assert len(traj) == 0
        assert traj.positions.shape == (0, 3)
        assert traj.timestamps.shape == (0,)

    def test_has_nan_false(self) -> None:
        traj = self._make_trajectory(3)
        assert traj.has_nan() is False

    def test_has_nan_true(self) -> None:
        p = Pose.identity()
        p.matrix[0, 0] = float("nan")
        traj = Trajectory(poses=[p], backend="dummy")
        assert traj.has_nan() is True

    def test_total_displacement(self) -> None:
        """沿 +x 移动 5 步, displacement = 4 * step."""
        from modules.projection.rotation import rotation_yaw

        poses = []
        for i in range(5):
            R = rotation_yaw(0)
            pos = np.array([i * 0.5, 0, 0])
            t = -R @ pos
            m = np.eye(4)
            m[:3, :3] = R
            m[:3, 3] = t
            poses.append(Pose(matrix=m, timestamp=float(i), frame_id=i))
        traj = Trajectory(poses=poses, backend="dummy")
        assert traj.total_displacement() == pytest.approx(2.0)  # 4 * 0.5

    def test_total_path_length(self) -> None:
        from modules.projection.rotation import rotation_yaw

        poses = []
        for i in range(5):
            R = rotation_yaw(0)
            pos = np.array([i * 0.5, 0, 0])
            t = -R @ pos
            m = np.eye(4)
            m[:3, :3] = R
            m[:3, 3] = t
            poses.append(Pose(matrix=m, timestamp=float(i), frame_id=i))
        traj = Trajectory(poses=poses, backend="dummy")
        assert traj.total_path_length() == pytest.approx(2.0)  # 4 * 0.5

    def test_summary(self) -> None:
        traj = self._make_trajectory(3)
        s = traj.summary()
        assert s["backend"] == "dummy"
        assert s["n_poses"] == 3
        assert s["has_nan"] is False
        assert "total_displacement" in s
        assert "total_path_length" in s
