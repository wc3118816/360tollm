"""modules.ingest.timesync 单测 (M005).

覆盖:
- align 找到最近邻 depth/pose
- 偏差超阈值时字段置 None + misalign_count++
- 无候选流时字段为 None
- 对齐后的 Frame 继承 primary 的 image/fps/source
- metadata.synced = True
"""

from __future__ import annotations

import numpy as np

from modules.ingest import Frame, TimeSync


def _make_frame(
    frame_id: int,
    timestamp: float,
    image_val: int = 100,
    depth_val: float | None = 2.0,
    pose_tx: float | None = 0.0,
) -> Frame:
    """造一个带 depth/pose 的 Frame."""
    img = np.full((2, 2, 3), image_val, dtype=np.uint8)
    depth = np.full((2, 2), depth_val, dtype=np.float32) if depth_val is not None else None
    pose = (
        np.array(
            [
                [1.0, 0.0, 0.0, pose_tx],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if pose_tx is not None
        else None
    )
    return Frame(
        frame_id=frame_id,
        timestamp=timestamp,
        image=img,
        fps=30.0,
        depth=depth,
        pose=pose,
        source="test",
    )


# ---------------- 基本对齐 ----------------


def test_align_finds_nearest_depth() -> None:
    """最近邻 depth 被选中."""
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 1.0, depth_val=None)  # 主流无 depth
    depth_cands = [
        _make_frame(10, 0.98, depth_val=1.5),
        _make_frame(11, 1.01, depth_val=2.0),  # 距 primary 0.01, 最近
        _make_frame(12, 1.05, depth_val=2.5),
    ]
    aligned = sync.align(primary, depth_candidates=depth_cands)
    assert aligned.depth is not None
    assert np.allclose(aligned.depth, 2.0)  # 选了 1.01 的那帧


def test_align_finds_nearest_pose() -> None:
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 2.0, pose_tx=None)
    pose_cands = [
        _make_frame(20, 1.95, pose_tx=0.5),
        _make_frame(21, 2.02, pose_tx=0.8),  # 距 primary 0.02, 最近
    ]
    aligned = sync.align(primary, pose_candidates=pose_cands)
    assert aligned.pose is not None
    assert aligned.pose[0, 3] == 0.8


def test_align_both_streams() -> None:
    """同时对齐 depth + pose."""
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 1.0, depth_val=None, pose_tx=None)
    depth_cands = [_make_frame(10, 1.01, depth_val=3.0)]
    pose_cands = [_make_frame(20, 1.02, pose_tx=1.5)]
    aligned = sync.align(primary, depth_candidates=depth_cands, pose_candidates=pose_cands)
    assert aligned.depth is not None
    assert aligned.pose is not None
    assert np.allclose(aligned.depth, 3.0)
    assert aligned.pose[0, 3] == 1.5


# ---------------- 偏差超阈值 ----------------


def test_align_exceeds_max_delta_returns_none() -> None:
    """最近邻偏差 > max_delta_sec 时, 该字段置 None."""
    sync = TimeSync(max_delta_sec=0.01)  # 严格阈值
    primary = _make_frame(1, 1.0, depth_val=None)
    depth_cands = [
        _make_frame(10, 0.95, depth_val=1.5),  # 距 primary 0.05 > 0.01
        _make_frame(11, 1.05, depth_val=2.0),  # 距 primary 0.05 > 0.01
    ]
    aligned = sync.align(primary, depth_candidates=depth_cands)
    assert aligned.depth is None
    assert sync.stats.misalign_count == 1


def test_misalign_count_accumulates() -> None:
    sync = TimeSync(max_delta_sec=0.01)
    primary = _make_frame(1, 1.0, depth_val=None, pose_tx=None)
    bad_cands = [_make_frame(10, 2.0, depth_val=1.0)]  # 偏差 1.0
    sync.align(primary, depth_candidates=bad_cands)
    sync.align(primary, pose_candidates=bad_cands)
    assert sync.stats.misalign_count == 2


# ---------------- 无候选流 ----------------


def test_align_no_candidates_returns_none() -> None:
    """无候选流时 depth/pose 为 None."""
    sync = TimeSync()
    primary = _make_frame(1, 1.0, depth_val=None, pose_tx=None)
    aligned = sync.align(primary)  # 不传候选
    assert aligned.depth is None
    assert aligned.pose is None


def test_align_empty_candidate_list_returns_none() -> None:
    sync = TimeSync()
    primary = _make_frame(1, 1.0, depth_val=None)
    aligned = sync.align(primary, depth_candidates=[])
    assert aligned.depth is None


# ---------------- Frame 继承 ----------------


def test_aligned_frame_inherits_primary_image() -> None:
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 1.0, image_val=200, depth_val=None)
    aligned = sync.align(primary, depth_candidates=[_make_frame(2, 1.01, depth_val=5.0)])
    # image 继承自 primary.
    assert aligned.image is not None
    assert aligned.image[0, 0, 0] == 200  # image_val=200


def test_aligned_frame_inherits_source_and_fps() -> None:
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 1.0, depth_val=None)
    primary_fps = primary.fps
    aligned = sync.align(primary, depth_candidates=[_make_frame(2, 1.01)])
    assert aligned.source == primary.source
    assert aligned.fps == primary_fps


def test_aligned_frame_metadata_synced_flag() -> None:
    sync = TimeSync()
    primary = _make_frame(1, 1.0)
    aligned = sync.align(primary)
    assert aligned.metadata.get("synced") is True


def test_aligned_frame_timestamp_preserved() -> None:
    """对齐后 timestamp 用 primary 的 (不变)."""
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 5.0, depth_val=None)
    aligned = sync.align(primary, depth_candidates=[_make_frame(2, 5.01)])
    assert aligned.timestamp == 5.0


# ---------------- 统计 ----------------


def test_stats_initial() -> None:
    sync = TimeSync(max_delta_sec=0.05)
    assert sync.stats.align_count == 0
    assert sync.stats.misalign_count == 0
    assert sync.stats.max_delta_sec == 0.05


def test_stats_align_count_increments() -> None:
    sync = TimeSync(max_delta_sec=0.05)
    primary = _make_frame(1, 1.0)
    sync.align(primary)
    sync.align(primary)
    assert sync.stats.align_count == 2
