"""M031 单元测试: ObjectObservation + TrackedObject 数据结构.

覆盖:
- ObjectObservation 构造 + 校验 (position shape / confidence)
- TrackedObject 构造 + 属性 (n_observations / first_seen / last_seen / current_position)
- add_observation + position_at + trajectory
- is_still + total_distance
- to_dict
- 空观测
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.spatial_memory import ObjectObservation, TrackedObject


def _make_obs(
    timestamp: float = 0.0,
    frame_id: int = 0,
    position: np.ndarray | None = None,
    confidence: float = 0.9,
) -> ObjectObservation:
    if position is None:
        position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return ObjectObservation(
        timestamp=timestamp,
        frame_id=frame_id,
        position=position,
        confidence=confidence,
    )


class TestObjectObservation:
    def test_basic_construction(self) -> None:
        obs = _make_obs()
        assert obs.timestamp == 0.0
        assert obs.frame_id == 0
        assert obs.confidence == 0.9

    def test_position_shape(self) -> None:
        with pytest.raises(ValueError, match="position"):
            ObjectObservation(timestamp=0, frame_id=0, position=np.array([1, 2]))

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_obs(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            _make_obs(confidence=-0.1)

    def test_to_dict(self) -> None:
        obs = _make_obs()
        d = obs.to_dict()
        assert d["timestamp"] == 0.0
        assert len(d["position"]) == 3


class TestTrackedObject:
    def test_empty(self) -> None:
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        assert tracked.n_observations == 0
        assert tracked.first_seen == 0.0
        assert tracked.last_seen == 0.0
        assert np.allclose(tracked.current_position, [0, 0, 0])

    def test_add_observation(self) -> None:
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        obs1 = _make_obs(timestamp=1.0, position=np.array([0, 0, 0], dtype=np.float32))
        obs2 = _make_obs(timestamp=2.0, position=np.array([1, 0, 0], dtype=np.float32))
        tracked.add_observation(obs1)
        tracked.add_observation(obs2)
        assert tracked.n_observations == 2
        assert tracked.first_seen == 1.0
        assert tracked.last_seen == 2.0
        assert np.allclose(tracked.current_position, [1, 0, 0])

    def test_add_observation_sorts_by_time(self) -> None:
        """观测自动按时间排序."""
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        obs_late = _make_obs(timestamp=2.0, position=np.array([1, 0, 0], dtype=np.float32))
        obs_early = _make_obs(timestamp=1.0, position=np.array([0, 0, 0], dtype=np.float32))
        tracked.add_observation(obs_late)  # 先加晚的.
        tracked.add_observation(obs_early)
        # 排序后: first=early, last=late.
        assert tracked.first_seen == 1.0
        assert tracked.last_seen == 2.0

    def test_position_at(self) -> None:
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        tracked.add_observation(
            _make_obs(timestamp=1.0, position=np.array([0, 0, 0], dtype=np.float32))
        )
        tracked.add_observation(
            _make_obs(timestamp=2.0, position=np.array([1, 0, 0], dtype=np.float32))
        )
        tracked.add_observation(
            _make_obs(timestamp=3.0, position=np.array([2, 0, 0], dtype=np.float32))
        )
        # 查询 t=1.5 → 最近的是 t=1.0.
        pos = tracked.position_at(1.5)
        assert np.allclose(pos, [0, 0, 0])
        # 查询 t=2.5 → 最近的是 t=2.0.
        pos = tracked.position_at(2.5)
        assert np.allclose(pos, [1, 0, 0])

    def test_trajectory(self) -> None:
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        tracked.add_observation(
            _make_obs(timestamp=1.0, position=np.array([0, 0, 0], dtype=np.float32))
        )
        tracked.add_observation(
            _make_obs(timestamp=2.0, position=np.array([1, 1, 0], dtype=np.float32))
        )
        traj = tracked.trajectory()
        assert traj.shape == (2, 3)
        assert np.allclose(traj[0], [0, 0, 0])
        assert np.allclose(traj[1], [1, 1, 0])

    def test_is_still(self) -> None:
        """静止判断."""
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        # 单次观测 → 静止.
        tracked.add_observation(
            _make_obs(timestamp=1.0, position=np.array([0, 0, 0], dtype=np.float32))
        )
        assert tracked.is_still() is True
        # 移动小 → 静止.
        tracked.add_observation(
            _make_obs(timestamp=2.0, position=np.array([0.05, 0, 0], dtype=np.float32))
        )
        assert tracked.is_still() is True
        # 移动大 → 非静止.
        tracked.add_observation(
            _make_obs(timestamp=3.0, position=np.array([1, 0, 0], dtype=np.float32))
        )
        assert tracked.is_still() is False

    def test_total_distance(self) -> None:
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        tracked.add_observation(
            _make_obs(timestamp=1.0, position=np.array([0, 0, 0], dtype=np.float32))
        )
        tracked.add_observation(
            _make_obs(timestamp=2.0, position=np.array([3, 0, 0], dtype=np.float32))
        )
        assert tracked.total_distance() == 3.0

    def test_to_dict(self) -> None:
        tracked = TrackedObject(tracked_id="T0001", label="chair")
        tracked.add_observation(_make_obs(timestamp=1.0))
        d = tracked.to_dict()
        assert d["tracked_id"] == "T0001"
        assert d["label"] == "chair"
        assert d["n_observations"] == 1
        assert len(d["observations"]) == 1
