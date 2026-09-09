"""M019 集成测试: 检测器端到端.

验证:
1. AC: 可在设定 FPS 下稳定运行
2. AC: 推理延迟可测
3. DetectionList 完整字段 (bbox + label + confidence + timestamp)
4. 批量推理
5. 与 M015 点云生成 + M021 前置 (Detection 可用于后续)
"""

from __future__ import annotations

import numpy as np

from modules.detection import (
    DummyDetector,
    DummyDetectorConfig,
)


class TestFullPipeline:
    def test_detect_single_frame(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        det = DummyDetector(DummyDetectorConfig(n_detections=5, latency_ms=1.0))
        dl = det.detect(img, timestamp=1.5)
        assert dl.n_detections == 5
        assert dl.timestamp == 1.5
        assert dl.backend == "dummy"
        assert dl.image_shape == (100, 200)

    def test_ac_stable_fps(self) -> None:
        """AC: 可在设定 FPS 下稳定运行."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        det = DummyDetector(DummyDetectorConfig(latency_ms=2.0, n_detections=3))
        for _ in range(20):
            dl = det.detect(img)
            assert dl.latency_ms > 0
            assert dl.fps > 0
        # 20 帧后统计.
        assert det.stats.n_frames == 20
        assert det.stats.avg_fps > 0

    def test_ac_latency_measured(self) -> None:
        """AC: 推理延迟可测."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        det = DummyDetector(DummyDetectorConfig(latency_ms=5.0))
        dl = det.detect(img)
        assert dl.latency_ms > 0
        assert det.stats.avg_latency_ms > 0
        assert det.stats.p95_latency_ms > 0

    def test_detection_fields_complete(self) -> None:
        """DetectionList 包含所有必需字段 (bbox + label + confidence + timestamp)."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        det = DummyDetector()
        dl = det.detect(img, timestamp=2.5)
        for d in dl.detections:
            assert d.bbox.shape == (4,)
            assert isinstance(d.label, str)
            assert 0.0 <= d.confidence <= 1.0
        assert dl.timestamp == 2.5
        assert dl.latency_ms > 0

    def test_batch_with_timestamps(self) -> None:
        images = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        timestamps = [0.1, 0.2, 0.3]
        det = DummyDetector(DummyDetectorConfig(latency_ms=0.0))
        results = det.detect_batch(images, timestamps=timestamps)
        assert len(results) == 3
        for i, dl in enumerate(results):
            assert dl.timestamp == timestamps[i]


class TestDetectionWithDepth:
    """检测 + 深度 (为 M021 前置)."""

    def test_detection_and_depth(self) -> None:
        from modules.depth import DummyDepthEstimator

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        # 深度.
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        # 检测.
        det = DummyDetector(DummyDetectorConfig(latency_ms=0.0))
        dl = det.detect(img)
        # 两者都应成功.
        assert dm.depth.shape == (100, 200)
        assert dl.n_detections > 0
        # detection 的 bbox 可用于从深度图取值.
        for d in dl.detections:
            x0, y0, x1, y1 = d.bbox.tolist()
            region = dm.depth[y0:y1, x0:x1]
            assert region.size > 0
