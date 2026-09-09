"""文件式视频摄入器 (离线回放).

用于 M005 replay 系统: 读本地 MP4/OSV 文件产出 Frame.
- 时间戳: 默认按文件 PTS; "ingest" 模式则用墙钟.
- 抽帧: 与 RTMPIngestor 共用 target_fps 节流逻辑.
- 离线数据集 (RealSee3D/Matterport3D) 的全景帧序列也走本 Ingestor 的变体
  (后续 M005 再加 DatasetIngestor 子类).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from modules.config import get_settings
from modules.config.settings import FrameCfg, IngestCfg
from modules.logging import get_logger

from .base import Ingestor
from .types import Frame

try:
    import av  # type: ignore

    _AV_AVAILABLE = True
except ImportError:  # pragma: no cover
    av = None  # type: ignore
    _AV_AVAILABLE = False


class FileIngestor(Ingestor):
    """从本地视频文件读帧的摄入器.

    与 RTMPIngestor 共享 PyAV 解码路径, 但:
        - 不重连 (文件不会"断线", EOF 即结束)
        - 默认按文件 PTS 提供时间戳, 便于回放对齐
        - 支持循环 (loop=True), 用于长时间稳定性测试
    """

    def __init__(
        self,
        path: str | Path,
        cfg: IngestCfg | None = None,
        frame_cfg: FrameCfg | None = None,
        loop: bool = False,
        throttle: bool = False,
    ) -> None:
        super().__init__(source_tag="file", cfg=cfg)
        if not _AV_AVAILABLE:
            raise ImportError(
                "PyAV is required for FileIngestor. Install with: uv sync --extra ingest"
            )
        self._path = Path(path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"video file not found: {self._path}")
        self._frame_cfg = frame_cfg or get_settings().frame
        self._loop = loop
        self._throttle = throttle
        self._container: Any | None = None
        self._video_stream: Any | None = None
        self._codec_ctx: Any | None = None
        self._last_emit_monotonic: float = 0.0
        self._frame_interval: float = 1.0 / max(self._cfg.target_fps, 0.1)
        self._demux_iter: Any = None
        self._log = get_logger("modules.ingest.file")
        self._log.info(
            "FileIngestor configured",
            path=str(self._path),
            target_fps=self._cfg.target_fps,
            loop=loop,
            throttle=throttle,
        )

    def describe_source(self) -> dict[str, Any]:
        return {
            "type": self._source_tag,
            "path": str(self._path),
            "target_fps": self._cfg.target_fps,
            "loop": self._loop,
            "throttle": self._throttle,
        }

    # ---------------- 子类接口 ----------------

    def _open_stream(self) -> None:
        try:
            self._container = av.open(str(self._path))
        except Exception as e:
            raise ConnectionError(f"av.open failed for {self._path}: {e}") from e
        try:
            self._video_stream = next(s for s in self._container.streams if s.type == "video")
        except StopIteration as e:
            raise ConnectionError(f"no video stream in {self._path}") from e
        self._codec_ctx = self._video_stream.codec_context
        # 关键: demux() 返回 generator, 每次调用都新建. 必须缓存复用,
        # 否则 _read_next_frame 每次都从头读 -> 死循环或重复读第一帧.
        self._demux_iter = self._container.demux(self._video_stream)
        self._log.info(
            "file stream opened",
            path=str(self._path),
            codec=self._codec_ctx.name,
            width=self._codec_ctx.width,
            height=self._codec_ctx.height,
            duration_sec=float(self._container.duration / 1e6) if self._container.duration else -1,
        )

    def _read_next_frame(self) -> Frame:
        if self._container is None or self._demux_iter is None:
            raise ConnectionError("container not opened")

        while True:
            try:
                packet = next(self._demux_iter)
            except StopIteration:
                # 文件 EOF.
                if self._loop:
                    self._log.info("file EOF, looping")
                    self._close_stream()
                    self._open_stream()
                    continue
                raise

            if packet.dts is None:
                self._record_drop("dts_is_none")
                continue

            try:
                frames = self._codec_ctx.decode(packet)
            except Exception as e:
                self._record_drop(f"decode_error:{e}")
                self._log.debug("decode skip", dts=packet.dts, error=str(e))
                continue
            if not frames:
                continue

            av_frame = frames[0]
            # 抽帧节流: 仅当 throttle=True 时按墙钟节流.
            # 离线回放默认 throttle=False, 全速产出所有解码帧,
            # 让上层 (M005 replay / M009 深度) 自行决定消费节奏.
            if self._throttle:
                now_mono = time.monotonic()
                elapsed = now_mono - self._last_emit_monotonic
                if self._last_emit_monotonic > 0 and elapsed < self._frame_interval:
                    self._record_drop("throttle_skip")
                    continue
                self._last_emit_monotonic = now_mono

            if self._frame_cfg.timestamp_source == "decode":
                pts = av_frame.pts
                time_base = (
                    float(self._video_stream.time_base) if self._video_stream.time_base else 0.0
                )
                timestamp = float(pts * time_base) if pts is not None else time.time()
            else:
                timestamp = time.time()

            img = _av_frame_to_rgb_ndarray(av_frame)

            # 文件实际帧率 (从 stream.average_rate, Fraction 形式, 转 float).
            source_fps = _stream_average_fps(self._video_stream)
            # throttle 时用 target_fps (产出节奏), 否则用源帧率 (全速产出).
            actual_fps = self._cfg.target_fps if self._throttle else source_fps

            return self._make_frame(
                image=img,
                timestamp=timestamp,
                fps=actual_fps,
                pts=int(av_frame.pts) if av_frame.pts is not None else -1,
                width=int(self._codec_ctx.width),
                height=int(self._codec_ctx.height),
                codec=str(self._codec_ctx.name),
                source_fps=source_fps,
            )

    def _close_stream(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception as e:  # noqa: BLE001
                self._log.debug("error closing container", error=str(e))
        self._container = None
        self._video_stream = None
        self._codec_ctx = None
        self._demux_iter = None


def _av_frame_to_rgb_ndarray(av_frame: Any) -> Any:
    """转 RGB ndarray, 优先 to_ndarray(format='rgb24')."""
    import numpy as np

    try:
        return av_frame.to_ndarray(format="rgb24")
    except Exception:  # noqa: BLE001
        # av_frame.to_image() 返回 PIL.Image; 无需显式 import PIL.
        img = av_frame.to_image()
        return np.array(img.convert("RGB"))


def _stream_average_fps(stream: Any) -> float:
    """从 PyAV stream.average_rate 提取 float 帧率.

    average_rate 是 fractions.Fraction; None 时回退到 0.0.
    """
    rate = getattr(stream, "average_rate", None)
    if rate is None:
        return 0.0
    try:
        return float(rate)
    except (TypeError, ValueError):
        return 0.0
