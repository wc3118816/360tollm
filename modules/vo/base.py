"""M011 Visual Odometry 基类 + 统计.

VisualOdometry ABC:
    estimate(frames) -> Trajectory     在帧序列上估计连续位姿

VOStats:
    累计 n_frames / n_success / failure_rate / total_latency
    供健康监控与 M044 延迟优化用.

状态机:
    IDLE -> RUNNING -> (对每帧: SUCCESS / FAIL)
                  -> DONE
                  -> FAILED (连续失败超阈值)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from modules.logging import get_logger

from .types import Pose, Trajectory

_LOG = get_logger("modules.vo")

# 连续失败超过此数则整体 FAILED.
_MAX_CONSECUTIVE_FAILS = 10


@dataclass
class VOStats:
    """Visual Odometry 运行统计."""

    n_frames: int = 0
    n_success: int = 0
    n_fail: int = 0
    total_latency_ms: float = 0.0
    consecutive_fails: int = 0
    backend: str = ""

    @property
    def failure_rate(self) -> float:
        """失败率 (0..1). AC: 失败率可统计."""
        return self.n_fail / self.n_frames if self.n_frames > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return 1.0 - self.failure_rate

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.n_frames if self.n_frames > 0 else 0.0

    def record_success(self, latency_ms: float) -> None:
        self.n_frames += 1
        self.n_success += 1
        self.total_latency_ms += latency_ms
        self.consecutive_fails = 0

    def record_fail(self, latency_ms: float = 0.0) -> None:
        self.n_frames += 1
        self.n_fail += 1
        self.total_latency_ms += latency_ms
        self.consecutive_fails += 1

    def is_failed(self) -> bool:
        """连续失败超阈值 → 整体 FAILED."""
        return self.consecutive_fails >= _MAX_CONSECUTIVE_FAILS

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_frames": self.n_frames,
            "n_success": self.n_success,
            "n_fail": self.n_fail,
            "failure_rate": round(self.failure_rate, 4),
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "consecutive_fails": self.consecutive_fails,
        }


class VisualOdometry(ABC):
    """Visual Odometry 抽象基类.

    子类必须实现:
        _estimate_poses(frames) -> list[Pose | None]   帧序列 → 位姿序列

    可选重写:
        describe() -> dict
        warmup() -> None
    """

    backend_name: str = "abstract"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._stats = VOStats(backend=self.backend_name)
        self._log = _LOG.bind(backend=self.backend_name, device=device)

    def estimate(self, frames: Sequence[Any]) -> Trajectory:
        """在帧序列上估计连续位姿.

        Args:
            frames: Frame 对象序列 (需有 .image / .timestamp / .frame_id).

        Returns:
            Trajectory. AC: 输出连续轨迹; 失败帧跳过 (不产生 Pose).
        """
        if len(frames) == 0:
            return Trajectory(poses=[], backend=self.backend_name)

        t0 = time.perf_counter()
        poses_or_none = self._estimate_poses(frames)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 统计成功/失败.
        valid_poses: list[Pose] = []
        for p in poses_or_none:
            if p is None:
                self._stats.record_fail(0.0)
            else:
                self._stats.record_success(latency_ms / max(len(poses_or_none), 1))
                valid_poses.append(p)

        # 确保至少有一个 pose (单位 pose 作为起点, 若无成功帧).
        if not valid_poses:
            self._log.warning("VO all frames failed; using identity as fallback")
            valid_poses.append(
                Pose.identity(
                    timestamp=getattr(frames[0], "timestamp", 0.0),
                    frame_id=getattr(frames[0], "frame_id", 0),
                )
            )

        traj = Trajectory(
            poses=valid_poses,
            backend=self.backend_name,
            metadata={
                "n_input_frames": len(frames),
                "n_valid_poses": len(valid_poses),
                "failure_rate": round(self._stats.failure_rate, 4),
                "total_latency_ms": round(latency_ms, 2),
            },
        )

        # AC: 轨迹无 NaN.
        if traj.has_nan():
            self._log.error("trajectory contains NaN", n_poses=len(traj))

        self._log.info(
            "VO estimate done",
            n_frames=len(frames),
            n_valid=len(valid_poses),
            failure_rate=round(self._stats.failure_rate, 4),
            latency_ms=round(latency_ms, 2),
        )
        return traj

    @abstractmethod
    def _estimate_poses(self, frames: Sequence[Any]) -> list[Pose | None]:
        """子类实现: 帧序列 → 位姿序列 (None = 该帧失败).

        首帧通常为单位 pose (世界原点).
        """

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "device": self.device}

    def warmup(self) -> None:  # noqa: B027
        """预热. 默认空实现."""

    @property
    def stats(self) -> VOStats:
        return self._stats
