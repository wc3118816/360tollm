"""M005 时间同步器: 多传感器流时间对齐.

场景: 当 RGB / Depth / Pose 来自不同源 (各自有独立时间戳), 时间戳可能不完全一致.
TimeSync 根据主流 (通常是 RGB) 的时间戳, 在从流中找最近邻帧, 输出对齐后的 Frame.

用法:
    from modules.ingest import TimeSync
    sync = TimeSync(max_delta_sec=0.05)  # 允许 50ms 偏差
    # 主流帧 + 候选从流帧列表
    aligned = sync.align(rgb_frame, depth_candidates=depth_frames, pose_candidates=pose_frames)

对齐策略: 最近邻 (nearest timestamp). 若从流时间戳与主流差 > max_delta_sec, 该流字段置 None.

注: DatasetIngestor 读的是 manifest 格式 (每帧 RGB/Depth/Pose 已预对齐), 通常不需 TimeSync.
TimeSync 主要服务未来场景: 多源实时流 (RGB from RTMP + Depth from USB + Pose from SLAM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.ingest.types import Frame
from modules.logging import get_logger


@dataclass(slots=True)
class SyncStats:
    """同步统计: 对齐次数 / 未对齐次数 / 最大偏差."""

    align_count: int = 0
    misalign_count: int = 0  # 从流时间戳偏差超阈值的次数
    max_delta_sec: float = 0.0


class TimeSync:
    """多流时间对齐器.

    主流通常是 RGB (从 Ingestor 产出). 从流 (depth / pose) 可能来自:
    - 独立传感器 (各自时间戳)
    - 预录制的序列 (按时间戳索引)

    对齐算法: 最近邻. 在从流候选帧中找时间戳最接近主流的帧.
    若最近邻偏差 > max_delta_sec, 该字段置 None (表示无对齐数据).
    """

    def __init__(self, max_delta_sec: float = 0.05) -> None:
        """
        Args:
            max_delta_sec: 允许的最大时间偏差 (秒). 超过则视为无对齐.
                默认 0.05s (50ms, 对应 20fps 流的 1 帧间隔).
        """
        self._max_delta = max_delta_sec
        self._stats = SyncStats(max_delta_sec=max_delta_sec)
        self._log = get_logger("modules.ingest.timesync")
        self._log.info("TimeSync configured", max_delta_sec=max_delta_sec)

    @property
    def stats(self) -> SyncStats:
        return self._stats

    def align(
        self,
        primary: Frame,
        depth_candidates: list[Frame] | None = None,
        pose_candidates: list[Frame] | None = None,
    ) -> Frame:
        """对齐从流到主流时间戳.

        Args:
            primary: 主流帧 (通常 RGB). 时间戳为基准.
            depth_candidates: 深度流候选帧列表 (已按时间排序).
            pose_candidates: 位姿流候选帧列表 (已按时间排序).

        Returns:
            新 Frame, image/fps/source/intrinsics 沿用 primary;
            depth/pose 从候选中取最近邻填充 (偏差超阈值则 None).
        """
        self._stats.align_count += 1
        ts = primary.timestamp

        depth = self._find_nearest(depth_candidates, ts, "depth") if depth_candidates else None
        pose = self._find_nearest(pose_candidates, ts, "pose") if pose_candidates else None

        # 构造对齐后的 Frame (继承 primary 的 image/fps/source, 填入对齐的 depth/pose).
        return Frame(
            frame_id=primary.frame_id,
            timestamp=ts,
            image=primary.image,
            fps=primary.fps,
            depth=depth,
            pose=pose,
            source=primary.source,
            intrinsics=dict(primary.intrinsics),
            metadata={**primary.metadata, "synced": True},
        )

    def _find_nearest(
        self,
        candidates: list[Frame] | None,
        target_ts: float,
        stream_name: str,
    ) -> Any:
        """在候选帧中找时间戳最接近 target_ts 的帧, 返回其 depth/pose 字段.

        若最近邻偏差 > max_delta_sec, 记录 misalign 并返回 None.
        """
        if not candidates:
            return None

        # 二分查找最近邻 (候选已按时间排序).
        import bisect

        timestamps = [f.timestamp for f in candidates]
        idx = bisect.bisect_left(timestamps, target_ts)

        # 比较 idx-1 和 idx 两个候选, 取更近的.
        best_idx = None
        best_delta = float("inf")
        for i in (idx - 1, idx):
            if 0 <= i < len(candidates):
                delta = abs(candidates[i].timestamp - target_ts)
                if delta < best_delta:
                    best_delta = delta
                    best_idx = i

        if best_idx is None:
            return None

        if best_delta > self._max_delta:
            self._stats.misalign_count += 1
            self._log.debug(
                "misaligned stream",
                stream=stream_name,
                target_ts=target_ts,
                nearest_ts=candidates[best_idx].timestamp,
                delta=best_delta,
                max_delta=self._max_delta,
            )
            return None

        # 返回对应字段 (depth 或 pose).
        field = "depth" if stream_name == "depth" else "pose"
        return getattr(candidates[best_idx], field)
