"""M019 单元测试: DummyDetector + DetectionStats.

覆盖:
- 基本检测 (返回 N 个 detection)
- bbox 在图像范围内
- label 来自配置列表
- confidence ∈ [conf_min, conf_max]
- 延迟可测 (latency_ms > 0)
- FPS 可计算
- 批量推理
- 统计累计 (n_frames / avg / p95)
- 可重复性 (固定 seed)
- AC: 设定 FPS 稳定运行
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.detection import (
    DetectionStats,
    DummyDetector,
    DummyDetectorConfig,
    create_detector,
)


def _make_image(h: int = 100, w: int = 200) -> np.ndarray:
    """生成测试图像."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDummyDetector:
    def test_basic_detect(self) -> None:
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        img = _make_image()
        dl = det.detect(img)
        assert dl.n_detections == 3
        assert dl.backend == "dummy"

    def test_bbox_in_range(self) -> None:
        det = DummyDetector(DummyDetectorConfig(n_detections=5, latency_ms=0))
        img = _make_image(100, 200)
        dl = det.detect(img)
        h, w = 100, 200
        for det_obj in dl.detections:
            assert det_obj.bbox[0] >= 0
            assert det_obj.bbox[1] >= 0
            assert det_obj.bbox[2] <= w
            assert det_obj.bbox[3] <= h
            assert det_obj.bbox[2] > det_obj.bbox[0]  # x_max > x_min
            assert det_obj.bbox[3] > det_obj.bbox[1]  # y_max > y_min

    def test_label_from_config(self) -> None:
        labels = ["cat", "dog", "bird"]
        det = DummyDetector(DummyDetectorConfig(n_detections=10, latency_ms=0, labels=labels))
        img = _make_image()
        dl = det.detect(img)
        for det_obj in dl.detections:
            assert det_obj.label in labels

    def test_confidence_range(self) -> None:
        det = DummyDetector(
            DummyDetectorConfig(n_detections=10, latency_ms=0, conf_min=0.3, conf_max=0.5)
        )
        img = _make_image()
        dl = det.detect(img)
        for det_obj in dl.detections:
            assert 0.3 <= det_obj.confidence <= 0.5

    def test_latency_measured(self) -> None:
        """AC: 推理延迟可测."""
        det = DummyDetector(DummyDetectorConfig(latency_ms=5.0))
        img = _make_image()
        dl = det.detect(img)
        assert dl.latency_ms > 0
        # 应接近 5ms (允许误差).
        assert dl.latency_ms >= 3.0

    def test_fps_calculated(self) -> None:
        """AC: FPS 可计算."""
        det = DummyDetector(DummyDetectorConfig(latency_ms=10.0))
        img = _make_image()
        dl = det.detect(img)
        assert dl.fps > 0
        # 10ms → 100 FPS.
        assert dl.fps <= 150.0

    def test_batch_detect(self) -> None:
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        images = [_make_image() for _ in range(3)]
        results = det.detect_batch(images)
        assert len(results) == 3
        for dl in results:
            assert dl.n_detections == 2

    def test_repeatability_with_seed(self) -> None:
        """固定 seed → 相同结果."""
        img = _make_image()
        det1 = DummyDetector(DummyDetectorConfig(seed=42, latency_ms=0))
        det2 = DummyDetector(DummyDetectorConfig(seed=42, latency_ms=0))
        dl1 = det1.detect(img)
        dl2 = det2.detect(img)
        for d1, d2 in zip(dl1.detections, dl2.detections, strict=True):
            assert np.array_equal(d1.bbox, d2.bbox)
            assert d1.label == d2.label


class TestDetectionStats:
    def test_record_and_avg(self) -> None:
        stats = DetectionStats(backend="dummy")
        stats.record(10.0)
        stats.record(20.0)
        assert stats.n_frames == 2
        assert stats.avg_latency_ms == 15.0  # (10+20)/2.

    def test_p95(self) -> None:
        stats = DetectionStats(backend="dummy")
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            stats.record(v)
        p95 = stats.p95_latency_ms
        # p95 应在 47-50 范围 (95th percentile).
        assert 40.0 <= p95 <= 50.0

    def test_avg_fps(self) -> None:
        stats = DetectionStats(backend="dummy")
        stats.record(10.0)  # 100 FPS.
        assert stats.avg_fps == 100.0

    def test_as_dict(self) -> None:
        stats = DetectionStats(backend="dummy")
        stats.record(10.0)
        d = stats.as_dict()
        assert d["backend"] == "dummy"
        assert d["n_frames"] == 1
        assert d["avg_latency_ms"] == 10.0


class TestACStableFPS:
    """AC: 可在设定 FPS 下稳定运行."""

    def test_stable_latency(self) -> None:
        """多次推理 → 延迟稳定."""
        det = DummyDetector(DummyDetectorConfig(latency_ms=5.0))
        img = _make_image()
        latencies = []
        for _ in range(10):
            dl = det.detect(img)
            latencies.append(dl.latency_ms)
        # 延迟应稳定 (std 不大).
        latencies_arr = np.array(latencies)
        assert np.std(latencies_arr) < 10.0  # 标准差 < 10ms.
        assert det.stats.n_frames == 10

    def test_meets_target_fps(self) -> None:
        """设定 latency → FPS 达标."""
        target_latency = 5.0  # 5ms → 200 FPS.
        det = DummyDetector(DummyDetectorConfig(latency_ms=target_latency))
        img = _make_image()
        dl = det.detect(img)
        # FPS = 1000 / latency.
        assert dl.fps >= 100.0  # 至少 100 FPS.


class TestCreateDetector:
    def test_create_dummy(self) -> None:
        det = create_detector("dummy")
        assert det.backend_name == "dummy"

    def test_create_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            create_detector("unknown")
