"""M031 单元测试: SpatialMemory 管理器.

覆盖:
- observe + 跨帧关联 (同 label + 距离近 → 同一 tracked_id)
- where_is (AC: 刚才那个物体现在在哪里)
- find_recent / find_by_label
- find_nearest_to_position
- cleanup (超时清除)
- 多帧连续观测
- 静态物体 → 同一 tracked_id
- 移动物体 → 新 tracked_id (距离远)
"""

from __future__ import annotations

import numpy as np

from modules.objects import Object3D, Object3DList
from modules.spatial_memory import SpatialMemory, SpatialMemoryConfig


def _make_object_list(
    objects: list[tuple[str, str, np.ndarray, float]],
    frame_id: int = 0,
    timestamp: float = 0.0,
) -> Object3DList:
    """构造 Object3DList: [(object_id, label, center, confidence), ...]."""
    objs = [
        Object3D(object_id=oid, label=label, center=center, confidence=conf)
        for oid, label, center, conf in objects
    ]
    return Object3DList(objects=objs, frame_id=frame_id, timestamp=timestamp)


class TestObserve:
    def test_observe_single_frame(self) -> None:
        ol = _make_object_list(
            [("obj_000_00", "chair", np.array([1, 0, 0], dtype=np.float32), 0.9)]
        )
        mem = SpatialMemory()
        ids = mem.observe(ol)
        assert len(ids) == 1
        assert mem.n_objects == 1

    def test_observe_multiple_frames_same_object(self) -> None:
        """静态物体多帧 → 同一 tracked_id."""
        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=1.0))
        # 帧 0: chair at (0, 0, 0).
        ol0 = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)],
            frame_id=0,
            timestamp=0.0,
        )
        ids0 = mem.observe(ol0)
        # 帧 1: chair at (0.1, 0, 0) → 距离 0.1 < 1.0 → 同一 ID.
        ol1 = _make_object_list(
            [("obj_001_00", "chair", np.array([0.1, 0, 0], dtype=np.float32), 0.9)],
            frame_id=1,
            timestamp=1.0,
        )
        ids1 = mem.observe(ol1)
        # 同一 tracked_id.
        assert ids0[0] == ids1[0]
        assert mem.n_objects == 1  # 仍一个物体.
        # 观测历史.
        tracked = mem.get_object(ids0[0])
        assert tracked is not None
        assert tracked.n_observations == 2

    def test_observe_moving_object_new_id(self) -> None:
        """移动物体距离远 → 新 tracked_id."""
        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=0.5))
        # 帧 0: chair at (0, 0, 0).
        ol0 = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)],
            frame_id=0,
            timestamp=0.0,
        )
        ids0 = mem.observe(ol0)
        # 帧 1: chair at (5, 0, 0) → 距离 5 > 0.5 → 新 ID.
        ol1 = _make_object_list(
            [("obj_001_00", "chair", np.array([5, 0, 0], dtype=np.float32), 0.9)],
            frame_id=1,
            timestamp=1.0,
        )
        ids1 = mem.observe(ol1)
        # 不同 tracked_id.
        assert ids0[0] != ids1[0]
        assert mem.n_objects == 2

    def test_different_labels_different_ids(self) -> None:
        """不同 label → 不同 tracked_id (即使位置近)."""
        mem = SpatialMemory()
        ol = _make_object_list(
            [
                ("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9),
                ("obj_000_01", "table", np.array([0.1, 0, 0], dtype=np.float32), 0.9),
            ]
        )
        ids = mem.observe(ol)
        assert len(ids) == 2
        assert ids[0] != ids[1]


class TestWhereIs:
    """AC: 刚才那个物体现在在哪里."""

    def test_where_is_basic(self) -> None:
        mem = SpatialMemory()
        ol = _make_object_list(
            [("obj_000_00", "chair", np.array([1, 2, 3], dtype=np.float32), 0.9)]
        )
        ids = mem.observe(ol)
        pos = mem.where_is(ids[0])
        assert pos is not None
        assert np.allclose(pos, [1, 2, 3])

    def test_where_is_after_move(self) -> None:
        """移动后 → 返回最新位置."""
        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=2.0))
        # 帧 0: at (0, 0, 0).
        ol0 = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)],
            frame_id=0,
            timestamp=0.0,
        )
        ids = mem.observe(ol0)
        # 帧 1: at (1, 0, 0) → 关联到同一 ID.
        ol1 = _make_object_list(
            [("obj_001_00", "chair", np.array([1, 0, 0], dtype=np.float32), 0.9)],
            frame_id=1,
            timestamp=1.0,
        )
        mem.observe(ol1)
        # where_is → 最新位置.
        pos = mem.where_is(ids[0])
        assert pos is not None
        assert np.allclose(pos, [1, 0, 0])

    def test_where_is_unknown(self) -> None:
        mem = SpatialMemory()
        assert mem.where_is("T9999") is None

    def test_where_is_empty_memory(self) -> None:
        mem = SpatialMemory()
        assert mem.where_is("T0001") is None


class TestFindRecent:
    def test_find_recent_by_label(self) -> None:
        mem = SpatialMemory()
        ol = _make_object_list(
            [
                ("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9),
                ("obj_000_01", "table", np.array([1, 0, 0], dtype=np.float32), 0.9),
            ],
            timestamp=5.0,
        )
        mem.observe(ol)
        recent = mem.find_recent(label="chair", time_window=10.0, current_time=5.0)
        assert len(recent) == 1
        assert recent[0].label == "chair"

    def test_find_recent_time_window(self) -> None:
        """时间窗口外的物体不返回."""
        mem = SpatialMemory(SpatialMemoryConfig(auto_cleanup=False))
        ol_old = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)], timestamp=0.0
        )
        mem.observe(ol_old)
        # 查询 current=20, window=10 → 阈值=10 → 物体 last_seen=0 < 10 → 不返回.
        recent = mem.find_recent(time_window=10.0, current_time=20.0)
        assert len(recent) == 0


class TestCleanup:
    def test_cleanup_removes_old(self) -> None:
        mem = SpatialMemory(SpatialMemoryConfig(max_age=5.0, auto_cleanup=False))
        ol_old = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)], timestamp=0.0
        )
        mem.observe(ol_old)
        assert mem.n_objects == 1
        # cleanup at t=10 → 物体 last_seen=0 < 10-5=5 → 清除.
        removed = mem.cleanup(10.0)
        assert removed == 1
        assert mem.n_objects == 0

    def test_cleanup_keeps_recent(self) -> None:
        mem = SpatialMemory(SpatialMemoryConfig(max_age=100.0, auto_cleanup=False))
        ol = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)], timestamp=5.0
        )
        mem.observe(ol)
        removed = mem.cleanup(10.0)
        assert removed == 0
        assert mem.n_objects == 1


class TestNearest:
    def test_find_nearest_to_position(self) -> None:
        mem = SpatialMemory()
        ol = _make_object_list(
            [
                ("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9),
                ("obj_000_01", "chair", np.array([5, 0, 0], dtype=np.float32), 0.9),
            ]
        )
        mem.observe(ol)
        result = mem.find_nearest_to_position(np.array([1, 0, 0]))
        assert result is not None
        tid, dist = result
        assert dist <= 1.0  # 离 (1,0,0) 最近的是 (0,0,0), 距离=1.


class TestSummary:
    def test_summary(self) -> None:
        mem = SpatialMemory()
        ol = _make_object_list(
            [
                ("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9),
                ("obj_000_01", "table", np.array([1, 0, 0], dtype=np.float32), 0.9),
            ]
        )
        mem.observe(ol)
        s = mem.summary()
        assert s["n_objects"] == 2
        assert s["n_total_observations"] == 2
        assert "chair" in s["labels"]
        assert "table" in s["labels"]

    def test_serialize(self) -> None:
        mem = SpatialMemory()
        ol = _make_object_list(
            [("obj_000_00", "chair", np.array([0, 0, 0], dtype=np.float32), 0.9)]
        )
        mem.observe(ol)
        json_str = mem.serialize()
        assert isinstance(json_str, str)
        assert "T0001" in json_str
