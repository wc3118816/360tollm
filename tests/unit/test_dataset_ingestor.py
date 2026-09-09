"""modules.ingest.dataset_ingestor 单测 (M005).

覆盖:
- DatasetIngestor 初始化 (manifest 不存在 / 空 manifest / 正常 manifest)
- describe_source 字段正确
- _read_next_frame 按顺序产出 Frame, depth/pose 就位
- EOF (loop=False) 抛 StopIteration
- loop=True 循环回放
- 文件缺失时 _record_drop + 字段置 None
- depth_unit mm → m 转换
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modules.config.settings import IngestCfg
from modules.ingest import DatasetIngestor, StreamStatus


def _make_test_dataset(
    root: Path,
    n_frames: int = 3,
    fps: float = 10.0,
    has_depth: bool = True,
    has_pose: bool = True,
) -> Path:
    """在临时目录造一个最小数据集."""
    from PIL import Image

    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    pose_dir = root / "pose"
    for d in (rgb_dir, depth_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    for i in range(n_frames):
        idx = f"{i:06d}"
        ts = i / fps

        # RGB: 2x2 红色.
        img = np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8)
        Image.fromarray(img, "RGB").save(rgb_dir / f"{idx}.png")

        if has_depth:
            depth = np.full((2, 2), 2000, dtype=np.uint16)  # 2000mm = 2m
            Image.fromarray(depth).save(depth_dir / f"{idx}.png")

        if has_pose:
            pose = {
                "world_to_camera": [
                    [1.0, 0.0, 0.0, -i * 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
            (pose_dir / f"{idx}.json").write_text(json.dumps(pose))

        frames_meta.append(
            {
                "frame_id": i + 1,
                "timestamp": ts,
                "rgb": f"rgb/{idx}.png",
                "depth": f"depth/{idx}.png" if has_depth else None,
                "pose": f"pose/{idx}.json" if has_pose else None,
            }
        )

    manifest = {
        "dataset_name": "test_dataset",
        "dataset_type": "synthetic",
        "fps": fps,
        "frame_count": n_frames,
        "width": 2,
        "height": 2,
        "has_depth": has_depth,
        "has_pose": has_pose,
        "depth_unit": "mm",
        "intrinsics_file": "intrinsics.json",
        "frames": frames_meta,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    intrinsics = {
        "type": "perspective",
        "fx": 320.0,
        "fy": 320.0,
        "cx": 1.0,
        "cy": 1.0,
        "width": 2,
        "height": 2,
    }
    (root / "intrinsics.json").write_text(json.dumps(intrinsics))

    return root


# ---------------- 初始化测试 ----------------


def test_dataset_ingestor_init_missing_manifest(tmp_path: Path) -> None:
    """manifest.json 不存在时抛 FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="manifest.json not found"):
        DatasetIngestor(tmp_path / "nonexistent")


def test_dataset_ingestor_init_empty_manifest(tmp_path: Path) -> None:
    """manifest frames 为空时, _open_stream 抛 ConnectionError."""
    (tmp_path / "manifest.json").write_text(json.dumps({"frames": []}))
    ing = DatasetIngestor(tmp_path)
    with pytest.raises(ConnectionError, match="manifest has no frames"):
        ing.start()


def test_dataset_ingestor_init_normal(tmp_path: Path) -> None:
    """正常初始化."""
    _make_test_dataset(tmp_path, n_frames=3)
    ing = DatasetIngestor(tmp_path)
    assert ing.frame_count == 0  # _open_stream 前还没读 manifest
    assert ing.describe_source()["dataset_root"] == str(tmp_path)


# ---------------- describe_source ----------------


def test_describe_source_fields(tmp_path: Path) -> None:
    _make_test_dataset(tmp_path, n_frames=5, fps=15.0)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    desc = ing.describe_source()
    assert desc["type"] == "dataset"
    assert desc["dataset_name"] == "test_dataset"
    assert desc["dataset_type"] == "synthetic"
    assert desc["fps"] == 15.0
    assert desc["frame_count"] == 5
    assert desc["loop"] is False
    ing.stop()


# ---------------- _read_next_frame ----------------


def test_read_frames_in_order(tmp_path: Path) -> None:
    _make_test_dataset(tmp_path, n_frames=3)
    ing = DatasetIngestor(tmp_path, cfg=IngestCfg(target_fps=10.0))
    ing.start()
    frames = list(ing.iter_frames())
    ing.stop()
    assert len(frames) == 3
    # frame_id 单调递增.
    ids = [f.frame_id for f in frames]
    assert ids == [1, 2, 3]
    # timestamp 单调.
    ts = [f.timestamp for f in frames]
    assert ts == [0.0, 0.1, 0.2]


def test_frame_has_rgb_image(tmp_path: Path) -> None:
    _make_test_dataset(tmp_path, n_frames=1)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.image is not None
    assert frame.image.shape == (2, 2, 3)
    assert frame.image.dtype == np.uint8


def test_frame_has_depth_in_meters(tmp_path: Path) -> None:
    """depth_unit=mm 时, 加载后转 float32 米单位."""
    _make_test_dataset(tmp_path, n_frames=1)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.depth is not None
    assert frame.depth.dtype == np.float32
    # 2000mm → 2.0m.
    assert np.allclose(frame.depth, 2.0)


def test_frame_has_pose_matrix(tmp_path: Path) -> None:
    _make_test_dataset(tmp_path, n_frames=1)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.pose is not None
    assert frame.pose.shape == (4, 4)
    # tx = 0 (首帧).
    assert frame.pose[0, 3] == 0.0


def test_frame_source_is_dataset(tmp_path: Path) -> None:
    _make_test_dataset(tmp_path, n_frames=1)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.source == "dataset"
    assert frame.metadata["dataset_name"] == "test_dataset"
    assert frame.metadata["dataset_type"] == "synthetic"


def test_frame_fps_from_manifest(tmp_path: Path) -> None:
    _make_test_dataset(tmp_path, n_frames=1, fps=25.0)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.fps == 25.0


# ---------------- EOF / loop ----------------


def test_eof_without_loop(tmp_path: Path) -> None:
    """读完所有帧后 status → STOPPED."""
    _make_test_dataset(tmp_path, n_frames=3)
    ing = DatasetIngestor(tmp_path, loop=False)
    ing.start()
    frames = list(ing.iter_frames())
    ing.stop()
    assert len(frames) == 3
    assert ing.status == StreamStatus.STOPPED


def test_loop_continues_after_eof(tmp_path: Path) -> None:
    """loop=True: 读完一遍后从头继续."""
    _make_test_dataset(tmp_path, n_frames=3)
    ing = DatasetIngestor(tmp_path, loop=True)
    ing.start()
    frames = list(ing.iter_frames(max_frames=7))  # 跨一遍 + 1 帧
    ing.stop()
    assert len(frames) == 7
    # frame_id 1..7 (跨循环不重置).
    assert [f.frame_id for f in frames] == list(range(1, 8))


# ---------------- 缺失流处理 ----------------


def test_missing_depth_returns_none(tmp_path: Path) -> None:
    """has_depth=False 时 frame.depth 为 None."""
    _make_test_dataset(tmp_path, n_frames=1, has_depth=False)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.depth is None


def test_missing_pose_returns_none(tmp_path: Path) -> None:
    """has_pose=False 时 frame.pose 为 None."""
    _make_test_dataset(tmp_path, n_frames=1, has_pose=False)
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.pose is None


def test_missing_rgb_file_records_drop(tmp_path: Path) -> None:
    """rgb 文件被删时, _record_drop + frame.image = None."""
    _make_test_dataset(tmp_path, n_frames=1)
    (tmp_path / "rgb" / "000000.png").unlink()  # 删掉
    ing = DatasetIngestor(tmp_path)
    ing.start()
    frame = next(ing.iter_frames(max_frames=1))
    ing.stop()
    assert frame.image is None
    assert ing.stats.frames_dropped > 0
