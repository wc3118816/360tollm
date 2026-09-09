"""M004 Frame Pipeline: 串联 Ingestor + 后处理中间件.

Pipeline 设计目标:
- 接收 Ingestor 产出的 Frame 流, 应用可配置的后处理中间件.
- 中间件签名: (Frame, ctx) -> Frame | None  (返回 None 表示丢弃该帧).
- 内置中间件: DedupFrames (帧去重, 基于 image hash) / KeyframeSelector (关键帧选择).
- 上层 (M005 replay / M009 深度 / M019 检测) 可直接消费 pipeline.iter_frames().

AC 覆盖:
- "支持可配置 FPS": Ingestor 已支持 target_fps; pipeline 透传.
- "时间戳单调": pipeline 不修改 timestamp, 由 Ingestor 保证.
- "丢帧可统计": pipeline 有自己的 stats (frames_in / frames_out / frames_filtered),
  与 Ingestor.stats.frames_dropped 叠加得到全局视图.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from modules.ingest.base import Ingestor
from modules.ingest.types import Frame
from modules.logging import get_logger

# 中间件签名: 接收 Frame + ctx dict, 返回 Frame (保留) / None (丢弃) / Frame (修改).
# ctx 可被中间件读写, 用于跨中间件传递状态 (如上一帧 hash).
Middleware = Callable[[Frame, dict[str, Any]], "Frame | None"]


@dataclass(slots=True)
class PipelineStats:
    """Pipeline 级统计, 与 IngestStats 叠加得到全局视图."""

    frames_in: int = 0  # 从 ingestor 接收的帧数
    frames_out: int = 0  # 产出到下游的帧数
    frames_filtered: int = 0  # 被中间件丢弃的帧数
    started_at: float = field(default_factory=time.monotonic)

    @property
    def drop_rate(self) -> float:
        """丢弃率 = filtered / in (无输入时 0)."""
        return self.frames_filtered / self.frames_in if self.frames_in > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "frames_filtered": self.frames_filtered,
            "drop_rate": self.drop_rate,
            "uptime_sec": time.monotonic() - self.started_at,
        }


class FramePipeline:
    """帧 pipeline: Ingestor → [middleware...] → 下游.

    用法:
        pipe = FramePipeline(ingestor, [DedupFrames(), KeyframeSelector(interval=2.0)])
        pipe.start()
        for frame in pipe.iter_frames(max_frames=100):
            ...
        pipe.stop()

    中间件按顺序应用; 任一中间件返回 None 则该帧被丢弃, 后续中间件不再处理.
    """

    def __init__(
        self,
        ingestor: Ingestor,
        middlewares: list[Middleware] | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._middlewares: list[Middleware] = list(middlewares or [])
        self._stats = PipelineStats()
        self._ctx: dict[str, Any] = {}  # 中间件共享上下文
        self._log = get_logger("modules.ingest.pipeline")
        self._log.info(
            "FramePipeline configured",
            n_middlewares=len(self._middlewares),
            middleware_names=[getattr(m, "__name__", str(m)) for m in self._middlewares],
        )

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    @property
    def ingestor_stats(self) -> dict[str, Any]:
        """底层 Ingestor 的 stats (frames_emitted / frames_dropped / ...)."""
        return self._ingestor.stats.as_dict()

    def add_middleware(self, mw: Middleware) -> None:
        self._middlewares.append(mw)
        self._log.info("middleware added", name=getattr(mw, "__name__", str(mw)))

    def start(self) -> None:
        self._ingestor.start()

    def stop(self) -> None:
        self._ingestor.stop()

    def iter_frames(self, max_frames: int | None = None) -> Iterator[Frame]:
        """迭代产出处理后的帧.

        Args:
            max_frames: 限制产出帧数 (经过中间件过滤后). None 表示不限.
        """
        emitted = 0
        for frame in self._ingestor.iter_frames():
            self._stats.frames_in += 1
            kept = self._apply_middlewares(frame)
            if kept is None:
                self._stats.frames_filtered += 1
                continue
            self._stats.frames_out += 1
            emitted += 1
            yield kept
            if max_frames is not None and emitted >= max_frames:
                self._log.info(
                    "pipeline reached max_frames",
                    max_frames=max_frames,
                    frames_in=self._stats.frames_in,
                    frames_filtered=self._stats.frames_filtered,
                )
                self.stop()
                return

    def _apply_middlewares(self, frame: Frame) -> Frame | None:
        """按顺序应用中间件; 任一返回 None 则停止.

        异常时记录日志并返回 None; frames_filtered 计数由调用方 iter_frames 统一处理,
        避免重复计数.
        """
        current: Frame | None = frame
        for mw in self._middlewares:
            if current is None:
                return None
            try:
                current = mw(current, self._ctx)
            except Exception as e:  # noqa: BLE001
                self._log.warning(
                    "middleware error, dropping frame",
                    middleware=getattr(mw, "__name__", str(mw)),
                    error=str(e),
                )
                return None
        return current


# ---------------- 内置中间件 ----------------


def DedupFrames(threshold: float = 0.5) -> Middleware:
    """帧去重中间件: 基于图像均值差的简单去重.

    连续两帧的灰度均值差 < threshold 时视为重复, 丢弃后者.
    适合静态场景下的冗余帧过滤; 计算成本极低.

    Args:
        threshold: 灰度均值差阈值 (0-255). 默认 0.5 (很敏感).
            调大 → 更宽松 (只去明显重复); 调小 → 更严格.
    """
    state = {"last_mean": None}

    def _dedup(frame: Frame, ctx: dict[str, Any]) -> Frame | None:
        import numpy as np

        if frame.image is None:
            return frame  # 无图不处理
        cur_mean = float(np.mean(frame.image))
        last = state["last_mean"]
        state["last_mean"] = cur_mean
        if last is None:
            return frame  # 首帧保留
        diff = abs(cur_mean - last)
        if diff < threshold:
            return None  # 重复, 丢弃
        return frame

    _dedup.__name__ = "DedupFrames"
    return _dedup


def KeyframeSelector(interval_sec: float = 2.0) -> Middleware:
    """关键帧选择中间件: 按时间间隔选关键帧.

    每隔 interval_sec 秒保留一帧, 其余丢弃.
    适合降低后续模块 (深度估计 / SLAM) 的计算负载.

    Args:
        interval_sec: 关键帧间隔 (秒). 默认 2.0 (0.5 fps).
    """
    state = {"last_keyframe_ts": -1.0}

    def _select(frame: Frame, ctx: dict[str, Any]) -> Frame | None:
        last = state["last_keyframe_ts"]
        if last < 0:
            state["last_keyframe_ts"] = frame.timestamp
            return frame  # 首帧必选
        if frame.timestamp - last >= interval_sec:
            state["last_keyframe_ts"] = frame.timestamp
            return frame
        return None

    _select.__name__ = "KeyframeSelector"
    return _select
