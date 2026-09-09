"""Ingestor 抽象基类.

所有具体摄入器 (RTMPIngestor / FileIngestor / 后续 DatasetIngestor) 都继承 Ingestor.

设计原则:
- 同步迭代器 API: ``for frame in ingestor.iter_frames(): ...``
  避免 asyncio 在数值栈 (numpy/torch) 中带来的复杂度.
- 流状态机显式, 通过 _set_status 维护, 配合 IngestStats 供健康监控.
- 断线重连: 子类实现 ``_open_stream()`` 与 ``_read_next_frame()``,
  基类在 _read_next_frame 抛 ConnectionError 时自动调度重连, 达到
  ``settings.ingest.reconnect_attempts`` 次后转 FAILED.
- 时间戳: 由 settings.frame.timestamp_source 决定 (ingest 墙钟 / decode PTS).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from modules.config import get_settings
from modules.config.settings import IngestCfg
from modules.logging import get_logger

from .types import Frame, IngestStats, StreamStatus


class Ingestor(ABC):
    """摄入器抽象基类.

    子类必须实现:
        _open_stream() -> None          打开/重连流, 失败抛 ConnectionError
        _read_next_frame() -> Frame      读下一帧, 失败抛 ConnectionError 或 StopIteration
        _close_stream() -> None         释放流资源

    可选重写:
        describe_source() -> dict       返回源描述信息 (URL/文件路径/数据集名等)
    """

    def __init__(self, source_tag: str, cfg: IngestCfg | None = None) -> None:
        self._source_tag = source_tag
        self._cfg = cfg or get_settings().ingest
        self._log = get_logger(f"modules.ingest.{self.__class__.__name__}")
        self._status = StreamStatus.IDLE
        self._stats = IngestStats()
        self._frame_seq = 0
        self._reconnect_attempts_made = 0
        self._stopped = False

    # ---------------- 公开 API ----------------

    @property
    def status(self) -> StreamStatus:
        return self._status

    @property
    def stats(self) -> IngestStats:
        return self._stats

    def describe_source(self) -> dict[str, Any]:
        """返回源描述. 子类可重写."""
        return {"type": self._source_tag}

    def start(self) -> None:
        """启动摄入: 打开流并切到 CONNECTING/STREAMING."""
        if self._status not in (StreamStatus.IDLE, StreamStatus.STOPPED):
            self._log.warning("start() called in non-idle status", status=self._status.value)
            return
        self._stopped = False
        self._set_status(StreamStatus.CONNECTING)
        try:
            self._open_stream()
            self._set_status(StreamStatus.STREAMING)
            self._log.info("stream started", **self.describe_source())
        except ConnectionError as e:
            self._handle_connection_failure(e)
            if self._status == StreamStatus.FAILED:
                raise

    def stop(self) -> None:
        """停止摄入, 释放资源. 幂等."""
        if self._status == StreamStatus.STOPPED:
            return
        self._stopped = True
        self._set_status(StreamStatus.STOPPED)
        try:
            self._close_stream()
        except Exception as e:  # noqa: BLE001
            self._log.warning("error during close_stream", error=str(e))
        self._log.info("ingestor stopped", **self._stats.as_dict())

    def iter_frames(self, max_frames: int | None = None) -> Iterator[Frame]:
        """主循环: 迭代产出 Frame.

        Args:
            max_frames: 可选, 限制总产出帧数 (测试/调试用). None 表示不限.

        Yields:
            Frame 对象.

        自动处理:
            - ConnectionError: 触发重连, 重连成功后继续产出; 失败达上限转 FAILED 后 return.
            - StopIteration: 流正常结束 (EOF), 转 STOPPED 后 return.
        """
        if self._status == StreamStatus.IDLE:
            self.start()
        if self._status == StreamStatus.FAILED:
            self._log.error("cannot iter_frames, status=FAILED", last_error=self._stats.last_error)
            return

        emitted = 0
        while not self._stopped:
            try:
                frame = self._read_next_frame()
            except StopIteration:
                self._log.info("stream EOF, stopping")
                self.stop()
                return
            except ConnectionError as e:
                self._handle_connection_failure(e)
                if self._status == StreamStatus.FAILED:
                    return
                # 重连成功则继续; 否则下一轮再读.
                continue
            except Exception as e:  # noqa: BLE001
                # 未预期异常: 记录但保持运行, 不让 pipeline 崩.
                self._log.error("unexpected error in read_next_frame", error=repr(e))
                self._stats.last_error = repr(e)
                continue

            self._stats.frames_emitted += 1
            self._stats.last_frame_at = time.monotonic()
            emitted += 1
            yield frame

            if max_frames is not None and emitted >= max_frames:
                self._log.info("reached max_frames, stopping", max_frames=max_frames)
                self.stop()
                return

    # ---------------- 子类接口 ----------------

    @abstractmethod
    def _open_stream(self) -> None:
        """打开流. 失败抛 ConnectionError."""

    @abstractmethod
    def _read_next_frame(self) -> Frame:
        """读下一帧. 流结束抛 StopIteration; 连接异常抛 ConnectionError."""

    @abstractmethod
    def _close_stream(self) -> None:
        """释放流资源."""

    # ---------------- 内部辅助 ----------------

    def _next_frame_id(self) -> int:
        self._frame_seq += 1
        return self._frame_seq

    def _make_frame(
        self,
        image: Any,
        timestamp: float | None = None,
        fps: float | None = None,
        **extra: Any,
    ) -> Frame:
        """构造 Frame, 自动填入 frame_id / source / timestamp / fps.

        Args:
            fps: 该帧对应的产出帧率. 默认用 self._cfg.target_fps.
                子类可在源流 fps 已知时覆盖 (如 FileIngestor 用文件实际帧率).
        """
        ts = timestamp if timestamp is not None else time.time()
        actual_fps = fps if fps is not None else self._cfg.target_fps
        return Frame(
            frame_id=self._next_frame_id(),
            timestamp=ts,
            image=image,
            fps=actual_fps,
            source=self._source_tag,
            intrinsics=self.describe_source(),
            metadata=extra,
        )

    def _record_drop(self, reason: str = "") -> None:
        """记录一次丢帧 (抽帧节流跳过 / 解码失败 / 时序异常等)."""
        self._stats.frames_dropped += 1
        if reason:
            self._log.debug(
                "frame dropped", reason=reason, total_dropped=self._stats.frames_dropped
            )

    def _set_status(self, status: StreamStatus) -> None:
        if self._status != status:
            self._log.info("status transition", frm=self._status.value, to=status.value)
        self._status = status

    def _handle_connection_failure(self, err: Exception) -> None:
        """处理 ConnectionError: 尝试重连, 失败达上限转 FAILED."""
        self._stats.last_error = str(err)
        self._set_status(StreamStatus.ERROR)
        self._log.warning(
            "connection failure",
            error=str(err),
            attempts_made=self._reconnect_attempts_made,
            attempts_max=self._cfg.reconnect_attempts,
        )
        if self._reconnect_attempts_made >= self._cfg.reconnect_attempts:
            self._set_status(StreamStatus.FAILED)
            self._log.error("reconnect attempts exhausted, FAILED", **self._stats.as_dict())
            return

        self._set_status(StreamStatus.RECONNECTING)
        self._reconnect_attempts_made += 1
        backoff = self._cfg.reconnect_backoff_sec * self._reconnect_attempts_made
        self._log.info("sleeping before reconnect", backoff_sec=backoff)
        time.sleep(backoff)

        try:
            self._open_stream()
            self._set_status(StreamStatus.STREAMING)
            self._stats.reconnect_count += 1
            self._log.info("reconnected", **self._stats.as_dict())
        except ConnectionError as e:
            self._stats.last_error = str(e)
            # 递归直到达到上限或成功.
            self._handle_connection_failure(e)
