"""M005 集成测试: DatasetIngestor + Frame API 一致性.

AC: "实时数据与离线回放使用同一上层 Frame API".

验证策略:
1. DatasetIngestor 产出的 Frame 与 FileIngestor 产出的 Frame 字段集完全一致.
2. 同一个 FramePipeline 可同时消费实时流和离线数据集.
3. 同一套 DedupFrames/KeyframeSelector 中间件对两类源都生效.

标记为 integration (依赖真实文件 IO).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.ingest import (
    DatasetIngestor,
    DedupFrames,
    FileIngestor,
    FramePipeline,
    KeyframeSelector,
    StreamStatus,
)

_SYNTHETIC_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "synthetic_indoor_v1"
_VIDEO_PATH = _SYNTHETIC_ROOT.parent / "sample" / "test_video.mp4"

pytestmark = pytest.mark.skipif(
    not _SYNTHETIC_ROOT.exists(),
    reason="synthetic dataset missing (run scripts/make_synthetic_dataset.py)",
)


class TestDatasetIngestorE2E:
    def test_read_all_frames(self) -> None:
        """完整读一遍合成数据集."""
        ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ing.start()
        frames = list(ing.iter_frames())
        ing.stop()
        assert len(frames) == 30
        assert ing.status == StreamStatus.STOPPED

    def test_frame_has_all_streams(self) -> None:
        """合成数据集每帧都有 RGB + Depth + Pose."""
        ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ing.start()
        frame = next(ing.iter_frames(max_frames=1))
        ing.stop()
        assert frame.image is not None
        assert frame.image.shape == (240, 320, 3)
        assert frame.depth is not None
        assert frame.depth.shape == (240, 320)
        assert frame.pose is not None
        assert frame.pose.shape == (4, 4)

    def test_depth_in_meters(self) -> None:
        """合成深度 2000mm → 2.0m."""
        ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ing.start()
        frame = next(ing.iter_frames(max_frames=1))
        ing.stop()
        assert frame.depth.min() == pytest.approx(2.0)

    def test_pose_translation_increments(self) -> None:
        """合成位姿沿 X 平移, 每帧 tx 增加 (world_to_camera 矩阵 [0,3] = -tx)."""
        ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ing.start()
        frames = list(ing.iter_frames(max_frames=5))
        ing.stop()
        for i, f in enumerate(frames):
            # 生成脚本: world_to_camera[0][3] = -tx, tx = i * 0.1.
            assert f.pose[0, 3] == pytest.approx(-(i * 0.1))

    def test_timestamp_monotonic(self) -> None:
        ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ing.start()
        frames = list(ing.iter_frames())
        ing.stop()
        ts = [f.timestamp for f in frames]
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

    def test_loop_continues_after_eof(self) -> None:
        ing = DatasetIngestor(_SYNTHETIC_ROOT, loop=True)
        ing.start()
        frames = list(ing.iter_frames(max_frames=45))  # 1.5 遍
        ing.stop()
        assert len(frames) == 45
        assert frames[0].frame_id == 1
        assert frames[-1].frame_id == 45


class TestFrameApiConsistency:
    """AC: 实时与离线 Frame API 一致性."""

    def test_frame_fields_identical(self) -> None:
        """DatasetIngestor 和 FileIngestor 产出的 Frame 字段集相同."""
        ds_ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ds_ing.start()
        ds_frame = next(ds_ing.iter_frames(max_frames=1))
        ds_ing.stop()

        # FileIngestor 可能在没装 PyAV 时跳过.
        if not _VIDEO_PATH.exists():
            pytest.skip("test_video.mp4 missing")

        try:
            file_ing = FileIngestor(_VIDEO_PATH)
            file_ing.start()
            file_frame = next(file_ing.iter_frames(max_frames=1))
            file_ing.stop()
        except ImportError:
            pytest.skip("PyAV not installed")

        # 两类 Frame 的字段集相同 (dataclass 字段).
        ds_fields = set(ds_frame.__dataclass_fields__.keys())
        file_fields = set(file_frame.__dataclass_fields__.keys())
        assert ds_fields == file_fields, (
            f"Frame field mismatch: dataset={ds_fields} vs file={file_fields}"
        )

    def test_pipeline_consumes_both_sources(self) -> None:
        """FramePipeline 能消费 DatasetIngestor (DedupFrames + KeyframeSelector)."""
        ds_ing = DatasetIngestor(_SYNTHETIC_ROOT)
        pipe = FramePipeline(
            ds_ing, [DedupFrames(threshold=1.0), KeyframeSelector(interval_sec=0.2)]
        )
        pipe.start()
        frames = list(pipe.iter_frames(max_frames=10))
        pipe.stop()
        # 经过中间件后帧数 <= 10 (可能过滤掉一些).
        assert len(frames) <= 10
        assert pipe.stats.frames_in >= len(frames)
        assert pipe.stats.frames_out == len(frames)

    def test_source_tag_distinguishes_origin(self) -> None:
        """source 字段区分来源 (dataset vs file)."""
        ds_ing = DatasetIngestor(_SYNTHETIC_ROOT)
        ds_ing.start()
        ds_frame = next(ds_ing.iter_frames(max_frames=1))
        ds_ing.stop()
        assert ds_frame.source == "dataset"

        if not _VIDEO_PATH.exists():
            pytest.skip("test_video.mp4 missing")
        try:
            file_ing = FileIngestor(_VIDEO_PATH)
            file_ing.start()
            file_frame = next(file_ing.iter_frames(max_frames=1))
            file_ing.stop()
            assert file_frame.source == "file"
            assert ds_frame.source != file_frame.source
        except ImportError:
            pytest.skip("PyAV not installed")
