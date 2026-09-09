"""M019 单元测试: Detection / DetectionList 数据结构.

覆盖:
- Detection 构造 + 校验 (bbox shape / confidence / bbox 有效性)
- 属性 (width / height / area / center)
- DetectionList 构造 + 属性 (n_detections / labels / fps)
- filter_by_confidence / filter_by_label
- summary / to_dict
- 空列表
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.detection import Detection, DetectionList


def _make_detection(
    bbox: list[int] | None = None,
    label: str = "person",
    confidence: float = 0.9,
) -> Detection:
    if bbox is None:
        bbox = [10, 20, 100, 200]
    return Detection(
        bbox=np.array(bbox, dtype=np.int32),
        label=label,
        label_id=0,
        confidence=confidence,
    )


class TestDetection:
    def test_basic_construction(self) -> None:
        det = _make_detection()
        assert det.label == "person"
        assert det.confidence == 0.9

    def test_bbox_shape(self) -> None:
        with pytest.raises(ValueError, match="bbox"):
            Detection(bbox=np.array([10, 20, 100], dtype=np.int32), label="x")

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_detection(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            _make_detection(confidence=-0.1)

    def test_invalid_bbox(self) -> None:
        # x_max <= x_min.
        with pytest.raises(ValueError, match="bbox"):
            _make_detection(bbox=[100, 20, 100, 200])
        # y_max <= y_min.
        with pytest.raises(ValueError, match="bbox"):
            _make_detection(bbox=[10, 200, 100, 200])

    def test_width_height(self) -> None:
        det = _make_detection(bbox=[10, 20, 100, 200])
        assert det.width == 90
        assert det.height == 180

    def test_area(self) -> None:
        det = _make_detection(bbox=[0, 0, 10, 20])
        assert det.area == 200

    def test_center(self) -> None:
        det = _make_detection(bbox=[0, 0, 10, 20])
        center = det.center
        assert center[0] == 5.0
        assert center[1] == 10.0

    def test_to_dict(self) -> None:
        det = _make_detection()
        d = det.to_dict()
        assert d["label"] == "person"
        assert d["width"] == 90
        assert "bbox" in d
        assert "center" in d


class TestDetectionList:
    def test_empty_list(self) -> None:
        dl = DetectionList()
        assert dl.n_detections == 0
        assert dl.labels == []

    def test_with_detections(self) -> None:
        d1 = _make_detection(label="person", confidence=0.9)
        d2 = _make_detection(label="chair", confidence=0.7)
        dl = DetectionList(detections=[d1, d2], latency_ms=10.0, backend="dummy")
        assert dl.n_detections == 2
        assert "person" in dl.labels
        assert "chair" in dl.labels

    def test_fps(self) -> None:
        dl = DetectionList(latency_ms=10.0)
        assert dl.fps == 100.0  # 1000/10.

    def test_fps_zero_latency(self) -> None:
        dl = DetectionList(latency_ms=0.0)
        assert dl.fps == 0.0

    def test_filter_by_confidence(self) -> None:
        d1 = _make_detection(confidence=0.9)
        d2 = _make_detection(confidence=0.3)
        dl = DetectionList(detections=[d1, d2])
        high = dl.filter_by_confidence(0.5)
        assert len(high) == 1

    def test_filter_by_label(self) -> None:
        d1 = _make_detection(label="person")
        d2 = _make_detection(label="chair")
        dl = DetectionList(detections=[d1, d2])
        people = dl.filter_by_label("person")
        assert len(people) == 1
        assert people[0].label == "person"

    def test_summary(self) -> None:
        d1 = _make_detection(label="person", confidence=0.9)
        d2 = _make_detection(label="chair", confidence=0.7)
        dl = DetectionList(
            detections=[d1, d2], latency_ms=10.0, backend="dummy", image_shape=(100, 200)
        )
        s = dl.summary()
        assert s["n_detections"] == 2
        assert s["fps"] == 100.0
        assert s["backend"] == "dummy"
        assert s["mean_confidence"] == 0.8  # (0.9+0.7)/2.
        assert s["image_shape"] == [100, 200]

    def test_to_dict(self) -> None:
        d1 = _make_detection(label="person")
        dl = DetectionList(detections=[d1], latency_ms=5.0)
        d = dl.to_dict()
        assert d["n_detections"] == 1
        assert len(d["detections"]) == 1

    def test_invalid_latency(self) -> None:
        with pytest.raises(ValueError, match="latency_ms"):
            DetectionList(latency_ms=-1.0)

    def test_invalid_image_shape(self) -> None:
        with pytest.raises(ValueError, match="image_shape"):
            DetectionList(image_shape=(100,))
