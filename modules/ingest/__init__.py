"""视频/图像摄入子包.

公开 API:
    Frame, IngestStats, StreamStatus   - 数据类型
    Ingestor                            - 抽象基类
    RTMPIngestor                        - RTMP/RTSP 实时流摄入
    FileIngestor                        - 本地视频文件回放
    FramePipeline, PipelineStats        - M004 帧 pipeline + 中间件
    DedupFrames, KeyframeSelector       - 内置中间件

使用示例:
    from modules.ingest import RTMPIngestor
    ingestor = RTMPIngestor("rtmp://example/live/stream")
    ingestor.start()
    for frame in ingestor.iter_frames(max_frames=100):
        print(frame.frame_id, frame.timestamp, frame.image.shape)
    ingestor.stop()

    # M004 pipeline 用法:
    from modules.ingest import FileIngestor, FramePipeline, DedupFrames, KeyframeSelector
    ing = FileIngestor("video.mp4")
    pipe = FramePipeline(ing, [DedupFrames(threshold=1.0), KeyframeSelector(interval_sec=2.0)])
    pipe.start()
    for frame in pipe.iter_frames(max_frames=100):
        ...
    pipe.stop()
"""

from __future__ import annotations

from .base import Ingestor
from .dataset_ingestor import DatasetIngestor
from .file_ingestor import FileIngestor
from .pipeline import (
    DedupFrames,
    FramePipeline,
    KeyframeSelector,
    PipelineStats,
)
from .rtmp_ingestor import RTMPIngestor
from .timesync import SyncStats, TimeSync
from .types import Frame, IngestStats, StreamStatus

__all__ = [
    "Frame",
    "IngestStats",
    "StreamStatus",
    "Ingestor",
    "RTMPIngestor",
    "FileIngestor",
    "DatasetIngestor",
    "FramePipeline",
    "PipelineStats",
    "DedupFrames",
    "KeyframeSelector",
    "TimeSync",
    "SyncStats",
]
