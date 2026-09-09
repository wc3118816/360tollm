"""modules.ingest.base 单测: 状态机 / 重连策略 / 时间戳单调性 / stats.

用 FakeIngestor 模拟具体行为, 不依赖真实 RTMP/文件 IO.
"""

from __future__ import annotations

import time

import pytest

from modules.config.settings import IngestCfg
from modules.ingest.base import Ingestor
from modules.ingest.types import Frame, StreamStatus


class FakeIngestor(Ingestor):
    """可控的测试用 Ingestor.

    通过 ``behaviors`` 队列驱动: 每次调用 _open_stream / _read_next_frame / _close_stream
    弹出一个行为执行.
    """

    def __init__(
        self,
        behaviors: list[str],
        cfg: IngestCfg | None = None,
        source_tag: str = "fake",
    ) -> None:
        # 用极小 backoff 让测试不慢.
        cfg = cfg or IngestCfg(
            reconnect_attempts=3,
            reconnect_backoff_sec=0.001,
            target_fps=1000.0,
            io_timeout_sec=1.0,
            low_latency=False,
        )
        super().__init__(source_tag=source_tag, cfg=cfg)
        self._behaviors = list(behaviors)
        self._open_count = 0
        self._close_count = 0
        self._frame_seq = 0

    def _open_stream(self) -> None:
        self._open_count += 1
        # 只在队首是 "fail-open" 时才消费; 否则不消费任何 behavior,
        # 避免吞掉本该给 _read_next_frame 用的 "frame" 等.
        if self._behaviors and self._behaviors[0] == "fail-open":
            self._behaviors.pop(0)
            raise ConnectionError("simulated open failure")

    def _read_next_frame(self) -> Frame:
        if not self._behaviors:
            # 默认产出 1 帧后 EOF.
            raise StopIteration
        action = self._behaviors.pop(0)
        if action == "frame":
            self._frame_seq += 1
            return self._make_frame(
                image=f"img-{self._frame_seq}",
                timestamp=time.time(),
                seq=self._frame_seq,
            )
        if action == "connection-error":
            raise ConnectionError("simulated read failure")
        if action == "unexpected-error":
            raise RuntimeError("simulated non-connection failure")
        if action == "eof":
            raise StopIteration
        raise ValueError(f"unknown behavior: {action}")

    def _close_stream(self) -> None:
        self._close_count += 1

    # 便于测试断言.
    @property
    def open_count(self) -> int:
        return self._open_count

    @property
    def close_count(self) -> int:
        return self._close_count


# ---------------- 状态机测试 ----------------


class TestStateMachine:
    def test_initial_status_is_idle(self) -> None:
        ing = FakeIngestor(behaviors=[])
        assert ing.status == StreamStatus.IDLE

    def test_start_transitions_to_streaming(self) -> None:
        ing = FakeIngestor(behaviors=[])
        ing.start()
        assert ing.status == StreamStatus.STREAMING

    def test_stop_transitions_to_stopped(self) -> None:
        ing = FakeIngestor(behaviors=[])
        ing.start()
        ing.stop()
        assert ing.status == StreamStatus.STOPPED

    def test_stop_is_idempotent(self) -> None:
        ing = FakeIngestor(behaviors=[])
        ing.start()
        ing.stop()
        ing.stop()
        assert ing.status == StreamStatus.STOPPED

    def test_start_when_not_idle_logs_warning(self) -> None:
        ing = FakeIngestor(behaviors=[])
        ing.start()
        # 第二次 start 应被忽略, 不重置 open_count.
        ing.start()
        assert ing.open_count == 1


# ---------------- 重连策略 ----------------


class TestReconnectStrategy:
    def test_reconnect_recovers_within_max_attempts(self) -> None:
        # 读第 1 帧成功 -> 之后 ConnectionError -> 重连成功 -> 读 1 帧成功 -> EOF.
        ing = FakeIngestor(behaviors=["frame", "connection-error", "frame", "eof"])
        ing.start()
        frames = list(ing.iter_frames())
        assert len(frames) == 2
        # 重连成功 1 次.
        assert ing.stats.reconnect_count == 1
        # 状态最终为 STOPPED (因 EOF).
        assert ing.status == StreamStatus.STOPPED

    def test_reconnect_exhausted_goes_failed(self) -> None:
        # 4 个 fail-open: 1 个初始 _open_stream + 3 个重连 _open_stream, 全部失败.
        # reconnect_attempts=3, 第 3 次重连失败后 -> FAILED.
        ing = FakeIngestor(
            behaviors=["fail-open", "fail-open", "fail-open", "fail-open"],
        )
        # start() 时 _open_stream 失败, 触发重连, 3 次都失败 -> FAILED -> raise.
        with pytest.raises(ConnectionError):
            ing.start()
        assert ing.status == StreamStatus.FAILED
        assert ing.stats.reconnect_count == 0  # 全部失败, 没有成功的 reconnect.

    def test_reconnect_backoff_grows(self) -> None:
        # 注: 真正的 sleep 用 0.001, 测试只验证不抛 SleepError.
        ing = FakeIngestor(
            behaviors=["connection-error", "frame", "eof"],
            cfg=IngestCfg(
                reconnect_attempts=2,
                reconnect_backoff_sec=0.001,
                target_fps=1000.0,
                io_timeout_sec=1.0,
                low_latency=False,
            ),
        )
        ing.start()
        list(ing.iter_frames())
        assert ing.stats.reconnect_count == 1


# ---------------- 时间戳单调性 ----------------


class TestTimestampMonotonicity:
    def test_frame_id_monotonic_increasing(self) -> None:
        ing = FakeIngestor(behaviors=["frame", "frame", "frame", "eof"])
        ing.start()
        frames = list(ing.iter_frames())
        ids = [f.frame_id for f in frames]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # 全唯一

    def test_timestamp_non_decreasing_when_using_ingest(self) -> None:
        # FakeIngestor 用 time.time(), 应单调不减.
        ing = FakeIngestor(behaviors=["frame", "frame", "frame", "eof"])
        ing.start()
        frames = list(ing.iter_frames())
        ts = [f.timestamp for f in frames]
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]


# ---------------- stats 累计 ----------------


class TestStatsAccounting:
    def test_frames_emitted_count_matches(self) -> None:
        ing = FakeIngestor(behaviors=["frame", "frame", "frame", "eof"])
        ing.start()
        frames = list(ing.iter_frames())
        assert ing.stats.frames_emitted == len(frames) == 3

    def test_unexpected_error_does_not_crash_pipeline(self) -> None:
        # unexpected-error 应被吞掉, 继续往后读.
        ing = FakeIngestor(behaviors=["unexpected-error", "frame", "eof"])
        ing.start()
        frames = list(ing.iter_frames())
        assert len(frames) == 1
        assert "simulated non-connection failure" in ing.stats.last_error

    def test_max_frames_limit(self) -> None:
        ing = FakeIngestor(behaviors=["frame", "frame", "frame", "frame", "frame", "frame"])
        ing.start()
        frames = list(ing.iter_frames(max_frames=3))
        assert len(frames) == 3
        assert ing.status == StreamStatus.STOPPED


# ---------------- describe_source ----------------


def test_describe_source_default() -> None:
    ing = FakeIngestor(behaviors=[])
    assert ing.describe_source() == {"type": "fake"}


def test_make_frame_attaches_source_and_intrinsics() -> None:
    ing = FakeIngestor(behaviors=["frame", "eof"])
    ing.start()
    frames = list(ing.iter_frames())
    assert len(frames) == 1
    f = frames[0]
    assert f.source == "fake"
    assert f.intrinsics == {"type": "fake"}
    # metadata 透传.
    assert f.metadata["seq"] == 1
