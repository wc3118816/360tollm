"""Ingest 模块通用类型与数据结构.

Frame 是后续所有 Mission (M004 解码/M005 回放/M009 深度/M019 检测 等)
消费的最小单元. 设计目标:
- 与具体来源 (RTMP / 文件 / 离线数据集) 解耦.
- 字段对齐 docs/device_ingestion_spec.md §3 MVP 输入方案.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StreamStatus(StrEnum):
    """Ingestor 流状态机.

    IDLE -> CONNECTING -> STREAMING -> (ERROR -> RECONNECTING -> CONNECTING)* -> STREAMING
                                                |-> FAILED (达到最大重连次数)
    """

    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(slots=True)
class Frame:
    """统一帧对象.

    Attributes:
        frame_id: 自增整数 ID, 单调递增, 由 Ingestor 维护.
        timestamp: 浮点秒. 默认墙钟时间, 也可由解码器 PTS 提供.
        image: RGB uint8 ndarray (H, W, 3). None 表示解码失败但帧槽位保留.
        fps: 该帧对应的产出帧率 (Hz). 来自 ingestor 的 target_fps 或源流 fps.
            M004 AC: 支持可配置 FPS; 该字段让下游知道采样节奏.
        depth: float32 深度图 (H, W), 米. M003 阶段恒为 None.
        pose: 4x4 world_to_camera SE3 矩阵. M003 阶段恒为 None.
        source: 来源标签 (rtmp / file / realsee3d / matterport3d / osmo360_offline).
        intrinsics: 相机内参描述. 全景用 {"type": "equirect", "width": W, "height": H}.
        metadata: 扩展字段, 任何附加键值对 (IMU/EXIF/codec 信息 等).
    """

    frame_id: int
    timestamp: float
    image: Any = None  # np.ndarray | None
    fps: float = 0.0
    depth: Any = None
    pose: Any = None
    source: str = "unknown"
    intrinsics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frame_id 必须 >= 0; 时间戳必须为正数 (0 表示未知, 允许但不常见).
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be >= 0, got {self.timestamp}")


@dataclass(slots=True)
class IngestStats:
    """Ingestor 运行统计, 供健康监控与 M006 frame pipeline 监控用."""

    frames_emitted: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    last_error: str = ""
    started_at: float = field(default_factory=time.monotonic)
    last_frame_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_emitted": self.frames_emitted,
            "frames_dropped": self.frames_dropped,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "last_frame_at": self.last_frame_at,
            "uptime_sec": time.monotonic() - self.started_at,
        }
