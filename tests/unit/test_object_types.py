"""M021 单元测试: Object3D / Object3DList 数据结构.

覆盖:
- Object3D 构造 + 校验 (center shape / bbox_3d shape / confidence)
- 属性 (to_dict)
- Object3DList 构造 + 属性 (n_objects / labels)
- filter_by_label / filter_by_confidence
- summary / to_dict
- 空列表
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.objects import Object3D, Object3DList


def _make_object(
    object_id: str = "chair_000_00",
    label: str = "chair",
    center: np.ndarray | None = None,
    confidence: float = 0.9,
) -> Object3D:
    if center is None:
        center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return Object3D(
        object_id=object_id,
        label=label,
        center=center,
        confidence=confidence,
    )


class TestObject3D:
    def test_basic_construction(self) -> None:
        obj = _make_object()
        assert obj.object_id == "chair_000_00"
        assert obj.label == "chair"
        assert obj.confidence == 0.9

    def test_center_shape(self) -> None:
        with pytest.raises(ValueError, match="center"):
            Object3D(object_id="x", label="x", center=np.array([1.0, 2.0]))

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_object(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            _make_object(confidence=-0.1)

    def test_bbox_3d_shape(self) -> None:
        with pytest.raises(ValueError, match="bbox_3d"):
            Object3D(
                object_id="x",
                label="x",
                center=np.zeros(3),
                bbox_3d=np.zeros((4, 3)),
            )

    def test_size_3d_shape(self) -> None:
        with pytest.raises(ValueError, match="size_3d"):
            Object3D(
                object_id="x",
                label="x",
                center=np.zeros(3),
                size_3d=np.array([1.0, 2.0]),
            )

    def test_to_dict(self) -> None:
        obj = _make_object()
        d = obj.to_dict()
        assert d["object_id"] == "chair_000_00"
        assert d["class"] == "chair"
        assert len(d["center"]) == 3
        assert len(d["bbox_3d"]) == 8
        assert len(d["size_3d"]) == 3


class TestObject3DList:
    def test_empty_list(self) -> None:
        ol = Object3DList()
        assert ol.n_objects == 0
        assert ol.labels == []

    def test_with_objects(self) -> None:
        o1 = _make_object(object_id="chair_000_00", label="chair")
        o2 = _make_object(object_id="person_000_01", label="person")
        ol = Object3DList(objects=[o1, o2], frame_id=0)
        assert ol.n_objects == 2
        assert "chair" in ol.labels
        assert "person" in ol.labels

    def test_filter_by_label(self) -> None:
        o1 = _make_object(label="chair")
        o2 = _make_object(label="person")
        ol = Object3DList(objects=[o1, o2])
        chairs = ol.filter_by_label("chair")
        assert len(chairs) == 1
        assert chairs[0].label == "chair"

    def test_filter_by_confidence(self) -> None:
        o1 = _make_object(confidence=0.9)
        o2 = _make_object(confidence=0.3)
        ol = Object3DList(objects=[o1, o2])
        high = ol.filter_by_confidence(0.5)
        assert len(high) == 1

    def test_summary(self) -> None:
        o1 = _make_object(label="chair", confidence=0.9)
        o2 = _make_object(label="person", confidence=0.7)
        ol = Object3DList(objects=[o1, o2], frame_id=5, timestamp=1.5)
        s = ol.summary()
        assert s["n_objects"] == 2
        assert s["frame_id"] == 5
        assert s["timestamp"] == 1.5
        assert s["mean_confidence"] == 0.8

    def test_to_dict(self) -> None:
        o1 = _make_object()
        ol = Object3DList(objects=[o1], frame_id=1)
        d = ol.to_dict()
        assert d["n_objects"] == 1
        assert len(d["objects"]) == 1
