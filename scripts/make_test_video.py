"""合成一段测试视频用于 M003 集成测试.

生成一个 10 秒、30fps、320x240、含帧号时间码的 MP4.
放到 datasets/sample/test_video.mp4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "datasets" / "sample" / "test_video.mp4"


def main() -> int:
    try:
        import av
    except ImportError:
        print("PyAV not installed. Run: uv sync --group dev", file=sys.stderr)
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        print(f"already exists: {OUTPUT}")
        return 0

    w, h, fps, duration_sec = 320, 240, 30, 10
    total_frames = fps * duration_sec

    container = av.open(str(OUTPUT), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "ultrafast"}

    for i in range(total_frames):
        # 每帧画一个变化的色块 + 帧号文本区域 (用简单像素填充代替文字).
        frame_array = np.zeros((h, w, 3), dtype=np.uint8)
        # 背景按帧渐变.
        bg_val = (i * 255 // total_frames) % 256
        frame_array[:, :, 0] = bg_val  # R
        frame_array[:, :, 1] = (bg_val * 2) % 256  # G
        frame_array[:, :, 2] = (255 - bg_val) % 256  # B
        # 中心方块随帧移动.
        cx = (i * 3) % w
        cy = h // 2
        frame_array[cy - 20 : cy + 20, cx - 20 : cx + 20] = [255, 255, 255]
        # 帧号区域 (左上角 80x20).
        frame_array[0:20, 0:80] = 0
        # 用简单方式编码帧号到像素 (供测试断言).
        # frame_array[0, 0] = [i & 0xFF, (i >> 8) & 0xFF, 0]

        av_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"created: {OUTPUT} ({w}x{h} {fps}fps {duration_sec}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
