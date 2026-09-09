"""modules.ingest.pipeline 单测 (M004).

覆盖:
- PipelineStats 初始化与 drop_rate 计算
- FramePipeline 串联中间件, frames_in/out/filtered 统计正确
- DedupFrames 帧去重: 首帧保留, 重复帧丢弃
- KeyframeSelector 关键帧选择: 按时间间隔保留
- 中间件异常时不崩, 该帧被丢弃
- 中间件返回 None 时不调用后续中间件
- pipeline + Ingestor stats 叠加
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from modules.ingest import (
    DedupFrames,
    Frame,
    FramePipeline,
    Ingestor,
    KeyframeSelector,
    PipelineStats,
    StreamStatus,
)

# ---------------- Fake Ingestor (用于 pipeline 单测) ----------------


class _FakeIngestor(Ingestor):
    """产出预设 Frame 序列的假 Ingestor, 供 pipeline 单测用."""

    def __init__(self, frames: list[Frame]) -> None:
        super().__init__(source_tag="fake")
        self._queue: list[Frame] = list(frames)

    def describe_source(self) -> dict[str, Any]:
        return {"type": "fake", "n_frames": len(self._queue)}

    def _open_stream(self) -> None:
        pass  # 无需打开

    def _read_next_frame(self) -> Frame:
        if not self._queue:
            raise StopIteration
        return self._queue.pop(0)

    def _close_stream(self) -> None:
        pass

    def start(self) -> None:
        # 重置 stats 与状态, 不真的 open 流.
        self._status = StreamStatus.STREAMING


def _make_frame(frame_id: int, timestamp: float, mean_val: int = 100) -> Frame:
    """造一个 4x4 RGB 帧, 全部像素 = mean_val."""
    img = np.full((4, 4, 3), mean_val, dtype=np.uint8)
    return Frame(
        frame_id=frame_id,
        timestamp=timestamp,
        image=img,
        fps=30.0,
        source="fake",
    )


# ---------------- PipelineStats ----------------


def test_pipeline_stats_initial() -> None:
    s = PipelineStats()
    assert s.frames_in == 0
    assert s.frames_out == 0
    assert s.frames_filtered == 0
    assert s.drop_rate == 0.0


def test_pipeline_stats_drop_rate() -> None:
    s = PipelineStats(frames_in=10, frames_filtered=3)
    s.frames_out = 7
    assert s.drop_rate == pytest.approx(0.3)


def test_pipeline_stats_drop_rate_zero_in() -> None:
    """无输入时 drop_rate = 0 (不除零)."""
    s = PipelineStats()
    assert s.drop_rate == 0.0


# ---------------- FramePipeline 基本流程 ----------------


def test_pipeline_passthrough_no_middleware() -> None:
    """无中间件: 所有帧透传, frames_in == frames_out."""
    frames = [_make_frame(i, float(i)) for i in range(1, 6)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 5
    assert pipe.stats.frames_in == 5
    assert pipe.stats.frames_out == 5
    assert pipe.stats.frames_filtered == 0


def test_pipeline_max_frames_limit() -> None:
    """max_frames 限制产出数."""
    frames = [_make_frame(i, float(i)) for i in range(1, 11)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [])
    pipe.start()
    out = list(pipe.iter_frames(max_frames=3))
    assert len(out) == 3
    assert pipe.stats.frames_out == 3
    assert pipe.stats.frames_in == 3  # 只消费了 3 帧就停


# ---------------- DedupFrames 中间件 ----------------


def test_dedup_first_frame_kept() -> None:
    """首帧必保留 (无前一帧可比)."""
    frames = [_make_frame(1, 1.0, mean_val=100)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [DedupFrames(threshold=1.0)])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 1
    assert pipe.stats.frames_filtered == 0


def test_dedup_identical_frames_filtered() -> None:
    """连续相同帧 → 后者被丢弃."""
    frames = [
        _make_frame(1, 1.0, mean_val=100),
        _make_frame(2, 2.0, mean_val=100),  # 与前帧相同, 丢
        _make_frame(3, 3.0, mean_val=100),  # 与前帧相同, 丢
    ]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [DedupFrames(threshold=1.0)])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 1  # 只保留首帧
    assert pipe.stats.frames_filtered == 2


def test_dedup_different_frames_kept() -> None:
    """不同帧 → 全保留."""
    frames = [
        _make_frame(1, 1.0, mean_val=50),
        _make_frame(2, 2.0, mean_val=100),
        _make_frame(3, 3.0, mean_val=200),
    ]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [DedupFrames(threshold=1.0)])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 3
    assert pipe.stats.frames_filtered == 0


# ---------------- KeyframeSelector 中间件 ----------------


def test_keyframe_first_frame_kept() -> None:
    """首帧必选."""
    frames = [_make_frame(1, 1.0)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [KeyframeSelector(interval_sec=2.0)])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 1


def test_keyframe_interval_filters() -> None:
    """按 2s 间隔选关键帧."""
    frames = [
        _make_frame(1, 0.0),  # 选 (首帧)
        _make_frame(2, 1.0),  # 距上 1s < 2s, 丢
        _make_frame(3, 2.0),  # 距上 2s >= 2s, 选
        _make_frame(4, 3.0),  # 距上 1s < 2s, 丢
        _make_frame(5, 4.0),  # 距上 2s >= 2s, 选
    ]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [KeyframeSelector(interval_sec=2.0)])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 3
    assert [f.frame_id for f in out] == [1, 3, 5]


# ---------------- 多中间件串联 ----------------


def test_pipeline_multiple_middlewares() -> None:
    """Dedup + Keyframe 串联, 按顺序应用."""
    frames = [
        _make_frame(1, 0.0, mean_val=50),
        _make_frame(2, 0.5, mean_val=50),  # dedup 丢 (相同)
        _make_frame(3, 1.0, mean_val=100),  # dedup 过, keyframe 丢 (距首 1s < 2s)
        _make_frame(4, 2.0, mean_val=200),  # dedup 过, keyframe 选 (距首 2s >= 2s)
    ]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [DedupFrames(threshold=1.0), KeyframeSelector(interval_sec=2.0)])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 2  # frame 1 (首) + frame 4
    assert [f.frame_id for f in out] == [1, 4]


# ---------------- 异常处理 ----------------


def test_pipeline_middleware_exception_drops_frame() -> None:
    """中间件抛异常时该帧被丢弃, pipeline 不崩."""

    def bad_mw(frame: Frame, ctx: dict[str, Any]) -> Frame | None:
        raise RuntimeError("boom")

    frames = [_make_frame(1, 1.0), _make_frame(2, 2.0), _make_frame(3, 3.0)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [bad_mw])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 0
    assert pipe.stats.frames_filtered == 3


def test_pipeline_middleware_return_none_stops_chain() -> None:
    """第一个中间件返回 None 后, 第二个不应被调用."""

    call_count = {"mw2": 0}

    def drop_all(frame: Frame, ctx: dict[str, Any]) -> Frame | None:
        return None

    def mw2(frame: Frame, ctx: dict[str, Any]) -> Frame | None:
        call_count["mw2"] += 1
        return frame

    frames = [_make_frame(1, 1.0), _make_frame(2, 2.0)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [drop_all, mw2])
    pipe.start()
    out = list(pipe.iter_frames())
    assert len(out) == 0
    assert call_count["mw2"] == 0  # mw2 从未被调用


# ---------------- Ingestor stats 叠加 ----------------


def test_pipeline_ingestor_stats_accessible() -> None:
    """pipeline.ingestor_stats 能访问底层 Ingestor 的 stats."""
    frames = [_make_frame(i, float(i)) for i in range(1, 4)]
    ing = _FakeIngestor(frames)
    pipe = FramePipeline(ing, [])
    pipe.start()
    list(pipe.iter_frames())
    stats_dict = pipe.ingestor_stats
    assert "frames_emitted" in stats_dict
    assert "frames_dropped" in stats_dict
    assert stats_dict["frames_emitted"] == 3
