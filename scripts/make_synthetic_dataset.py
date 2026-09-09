"""生成合成离线数据集 (RGB + Depth + Pose) 用于 M005 测试.

输出目录结构符合 docs/dataset_format_spec.md:
    datasets/synthetic_indoor_v1/
    ├── manifest.json
    ├── rgb/000000.png ... 000029.png
    ├── depth/000000.png ... 000029.png
    ├── pose/000000.json ... 000029.json
    └── intrinsics.json

合成场景: 相机沿 X 轴匀速平移, 30 帧 @10fps.
RGB: 渐变色块, 帧间有微小变化 (模拟运动视差).
Depth: 简单平面深度 (前方 2m 处一面墙).
Pose: 4x4 矩阵, 沿 X 平移 0.1m/帧.

用法:
    uv run python scripts/make_synthetic_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path("datasets/synthetic_indoor_v1")
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    pose_dir = root / "pose"
    for d in (rgb_dir, depth_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_frames = 30
    fps = 10.0
    width, height = 320, 240

    # intrinsics: 简单针孔模型.
    intrinsics = {
        "type": "perspective",
        "fx": 320.0,
        "fy": 320.0,
        "cx": width / 2,
        "cy": height / 2,
        "width": width,
        "height": height,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    (root / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    frames_manifest = []
    for i in range(n_frames):
        idx_str = f"{i:06d}"
        timestamp = i / fps

        # RGB: 渐变色块, 帧间偏移 (模拟运动).
        img = np.zeros((height, width, 3), dtype=np.uint8)
        offset = i * 5  # 每帧右移 5 像素.
        for y in range(height):
            for x in range(width):
                r = (x + offset) % 256
                g = (y + i * 3) % 256
                b = 128
                img[y, x] = [r, g, b]
        Image.fromarray(img, "RGB").save(rgb_dir / f"{idx_str}.png")

        # Depth: 平面深度 2000mm (2m), 全图一致.
        depth = np.full((height, width), 2000, dtype=np.uint16)
        Image.fromarray(depth).save(depth_dir / f"{idx_str}.png")

        # Pose: 沿 X 轴平移 0.1m/帧.
        tx = i * 0.1
        pose = {
            "world_to_camera": [
                [1.0, 0.0, 0.0, -tx],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
        (pose_dir / f"{idx_str}.json").write_text(json.dumps(pose, indent=2))

        frames_manifest.append(
            {
                "frame_id": i + 1,
                "timestamp": timestamp,
                "rgb": f"rgb/{idx_str}.png",
                "depth": f"depth/{idx_str}.png",
                "pose": f"pose/{idx_str}.json",
            }
        )

    manifest = {
        "dataset_name": "synthetic_indoor_v1",
        "dataset_type": "synthetic",
        "fps": fps,
        "frame_count": n_frames,
        "width": width,
        "height": height,
        "has_depth": True,
        "has_pose": True,
        "depth_unit": "mm",
        "intrinsics_file": "intrinsics.json",
        "frames": frames_manifest,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"synthetic dataset created: {root} ({n_frames} frames)")


if __name__ == "__main__":
    main()
