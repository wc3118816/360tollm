"""RTMP/RTSP/HTTP 视频流摄入器.

基于 PyAV (libav 绑定), 支持任意 ffmpeg 可处理协议 (rtmp:// / rtsp:// / http:// / file://).
设计:
- 解码主循环用 PyAV 的 Container / stream.video.
- 抽帧: 按目标 FPS 节流, 低于源帧率时跳帧 (默认 2fps, 适配后续 LLM 推理节奏).
- 时间戳: settings.frame.timestamp_source == "decode" 时用 PTS, 否则用墙钟 time.time().
- 断线重连: 基类已实现; 本类 _open_stream 抛 ConnectionError 即可触发.

参考: aircast (https://github.com/getrismine/aircast) DJI 直播接收用类似架构.
"""

from __future__ import annotations

import time
from typing import Any

from modules.config import get_settings
from modules.config.settings import FrameCfg, IngestCfg
from modules.logging import get_logger

from .base import Ingestor
from .types import Frame

# PyAV 是可选依赖, 仅在真正需要解码时才 import.
try:
    import av  # type: ignore

    _AV_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境差异
    av = None  # type: ignore
    _AV_AVAILABLE = False


class RTMPIngestor(Ingestor):
    """通用 RTMP/RTSP 视频流摄入器.

    不绑定具体相机 (DJI Osmo 360 当前无公开全景流, 见 device_ingestion_spec.md §1.2).
    主要服务:
        1) 未来外置 360 摄像头 (Ricoh Theta / Insta360 Pro 等) 的 RTMP/RTSP 推流
        2) MediaMTX / nginx-rtmp 等中转服务器转发
        3) 测试用本地 MP4 文件 (file:// 或直接路径)
    """

    def __init__(
        self,
        rtmp_url: str,
        cfg: IngestCfg | None = None,
        frame_cfg: FrameCfg | None = None,
    ) -> None:
        super().__init__(source_tag="rtmp", cfg=cfg)
        if not _AV_AVAILABLE:
            raise ImportError(
                "PyAV is required for RTMPIngestor. Install with: uv sync --extra ingest"
            )
        if not rtmp_url:
            raise ValueError("rtmp_url must not be empty")
        self._rtmp_url = rtmp_url
        self._frame_cfg = frame_cfg or get_settings().frame
        self._container: Any | None = None
        self._video_stream: Any | None = None
        self._codec_ctx: Any | None = None
        self._last_emit_monotonic: float = 0.0
        self._frame_interval: float = 1.0 / max(self._cfg.target_fps, 0.1)
        self._demux_iter: Any = None
        self._log = get_logger("modules.ingest.rtmp")
        self._log.info(
            "RTMPIngestor configured",
            url=rtmp_url,
            target_fps=self._cfg.target_fps,
            frame_interval=self._frame_interval,
            timestamp_source=self._frame_cfg.timestamp_source,
        )

    def describe_source(self) -> dict[str, Any]:
        return {
            "type": self._source_tag,
            "url": self._rtmp_url,
            "target_fps": self._cfg.target_fps,
        }

    # ---------------- 子类接口实现 ----------------

    def _open_stream(self) -> None:
        """打开 PyAV Container. 失败抛 ConnectionError."""
        try:
            # options 提升 RTMP 鲁棒性.
            timeout_us = (
                int(self._cfg.io_timeout_sec * 1_000_000)
                if self._cfg.io_timeout_sec > 0
                else 5_000_000
            )
            options: dict[str, str] = {
                "rw_timeout": str(timeout_us),
                "timeout": str(timeout_us),
                "max_delay": str(timeout_us // 10),
            }
            if self._cfg.low_latency:
                options["fflags"] = "nobuffer"
                options["flags"] = "low_delay"
            self._container = av.open(self._rtmp_url, options=options)
        except Exception as e:
            # 任何 av.open 失败统一转 ConnectionError, 触发基类重连.
            raise ConnectionError(f"av.open failed for {self._rtmp_url}: {e}") from e

        # 选第一个视频流.
        try:
            self._video_stream = next(s for s in self._container.streams if s.type == "video")
        except StopIteration as e:
            raise ConnectionError(f"no video stream in {self._rtmp_url}") from e
        self._codec_ctx = self._video_stream.codec_context
        # 关键: demux() 返回 generator, 每次调用都新建. 必须缓存复用,
        # 否则 _read_next_frame 每次都从头读 -> 死循环或重复读第一帧.
        self._demux_iter = self._container.demux(self._video_stream)
        self._log.info(
            "stream opened",
            url=self._rtmp_url,
            codec=self._codec_ctx.name,
            width=self._codec_ctx.width,
            height=self._codec_ctx.height,
            pix_fmt=self._codec_ctx.pix_fmt,
        )

    def _read_next_frame(self) -> Frame:
        """从容器读下一帧并解码. 抽帧按 target_fps 节流."""
        if self._container is None or self._demux_iter is None:
            raise ConnectionError("container not opened")

        while True:
            try:
                packet = next(self._demux_iter)
            except StopIteration:
                # demux 迭代器耗尽 = 流结束.
                raise
            except Exception as e:
                # 任何 IO/解码异常视为连接故障, 触发重连.
                raise ConnectionError(f"demux/read error: {e}") from e

            if packet.dts is None:
                # 时序信息缺失, 跳过 (常见于流头/尾).
                self._record_drop("dts_is_none")
                continue

            try:
                frames = self._codec_ctx.decode(packet)
            except Exception as e:
                # 解码错误通常可恢复, 跳过该包.
                self._record_drop(f"decode_error:{e}")
                self._log.debug("decode skip", dts=packet.dts, error=str(e))
                continue
            if not frames:
                continue

            av_frame = frames[0]
            # 抽帧节流: 用墙钟 monotonic, 避免 PTS 跳变导致抖动.
            now_mono = time.monotonic()
            elapsed = now_mono - self._last_emit_monotonic
            if self._last_emit_monotonic > 0 and elapsed < self._frame_interval:
                self._record_drop("throttle_skip")
                continue
            self._last_emit_monotonic = now_mono

            # 时间戳.
            if self._frame_cfg.timestamp_source == "decode":
                # PTS 以 stream.time_base 为单位; 转 秒.
                pts = av_frame.pts
                time_base = (
                    float(self._video_stream.time_base) if self._video_stream.time_base else 0.0
                )
                timestamp = float(pts * time_base) if pts is not None else time.time()
            else:
                timestamp = time.time()

            # 转 ndarray (BGR->RGB); PyAV 的 to_ndarray 默认返回 numpy array.
            # 注: av_frame.to_ndarray() 返回 (H, W, 3) uint8, 但格式可能为 yuv420p;
            # 我们让 codec_ctx 自动转换到 rgb24.
            img = _av_frame_to_rgb_ndarray(av_frame, self._codec_ctx)

            return self._make_frame(
                image=img,
                timestamp=timestamp,
                pts=int(av_frame.pts) if av_frame.pts is not None else -1,
                width=int(self._codec_ctx.width),
                height=int(self._codec_ctx.height),
                codec=str(self._codec_ctx.name),
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


# ---------------- 辅助函数 ----------------


def _av_frame_to_rgb_ndarray(av_frame: Any, codec_ctx: Any) -> Any:
    """把 PyAV frame 转 RGB uint8 ndarray.

    优先用 av_frame.to_ndarray(format="rgb24"), 失败回退到重新 decode.
    """
    import numpy as np

    try:
        return av_frame.to_ndarray(format="rgb24")
    except Exception:  # noqa: BLE001
        # 兜底: 用 Pillow/reformat.

        img = av_frame.to_image()
        return np.array(img.convert("RGB"))
