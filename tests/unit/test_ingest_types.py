"""modules.ingest.types 单测."""

from __future__ import annotations

import pytest

from modules.ingest.types import Frame, IngestStats, StreamStatus


def test_frame_minimal_construction() -> None:
    f = Frame(frame_id=1, timestamp=1000.0)
    assert f.frame_id == 1
    assert f.timestamp == 1000.0
    assert f.image is None
    assert f.source == "unknown"
    assert f.intrinsics == {}
    assert f.metadata == {}


def test_frame_full_construction() -> None:
    f = Frame(
        frame_id=2,
        timestamp=2000.0,
        image=b"fake-image",
        fps=30.0,
        depth=b"fake-depth",
        pose=b"fake-pose",
        source="rtmp",
        intrinsics={"type": "equirect", "width": 7680, "height": 3840},
        metadata={"pts": 12345},
    )
    assert f.source == "rtmp"
    assert f.fps == 30.0
    assert f.intrinsics["type"] == "equirect"
    assert f.metadata["pts"] == 12345


def test_frame_fps_default_zero() -> None:
    """M004 AC: fps 字段默认 0.0 (未指定时)."""
    f = Frame(frame_id=1, timestamp=0.0)
    assert f.fps == 0.0


def test_frame_id_negative_rejected() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        Frame(frame_id=-1, timestamp=0.0)


def test_frame_timestamp_negative_rejected() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        Frame(frame_id=0, timestamp=-0.1)


def test_frame_zero_values_allowed() -> None:
    # frame_id=0 与 timestamp=0 都允许 (边界值).
    f = Frame(frame_id=0, timestamp=0.0)
    assert f.frame_id == 0
    assert f.timestamp == 0.0


def test_stream_status_is_str_enum() -> None:
    assert StreamStatus.IDLE == "idle"
    assert StreamStatus.STREAMING.value == "streaming"
    # 可比较字符串.
    assert StreamStatus.FAILED != StreamStatus.STREAMING


def test_ingest_stats_initial() -> None:
    stats = IngestStats()
    assert stats.frames_emitted == 0
    assert stats.frames_dropped == 0
    assert stats.reconnect_count == 0
    assert stats.last_error == ""
    assert stats.last_frame_at == 0.0


def test_ingest_stats_as_dict() -> None:
    stats = IngestStats(frames_emitted=10, reconnect_count=2)
    d = stats.as_dict()
    assert d["frames_emitted"] == 10
    assert d["reconnect_count"] == 2
    assert "uptime_sec" in d
    assert d["uptime_sec"] >= 0


def test_ingest_stats_uptime_grows() -> None:
    import time

    stats = IngestStats()
    t1 = stats.as_dict()["uptime_sec"]
    time.sleep(0.05)
    t2 = stats.as_dict()["uptime_sec"]
    assert t2 > t1, f"uptime did not grow: {t1} -> {t2}"


def test_ingest_stats_frames_dropped_accumulates() -> None:
    """M004 AC: 丢帧可统计 — frames_dropped 字段可累计."""
    stats = IngestStats()
    assert stats.frames_dropped == 0
    stats.frames_dropped += 3
    stats.frames_dropped += 2
    assert stats.frames_dropped == 5
    assert stats.as_dict()["frames_dropped"] == 5
