"""M009 单元测试: DepthMap 数据结构.

覆盖:
- 构造 + shape 推断
- depth/confidence 类型与形状校验
- confidence 截断到 [0, 1]
- latency_ms 负值拒绝
- to_dict / summary
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.depth import DepthMap


class TestDepthMapConstruction:
    def test_basic_construction(self) -> None:
        depth = np.full((10, 20), 5.0, dtype=np.float32)
        conf = np.full((10, 20), 0.9, dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf, latency_ms=42.0, backend="dummy")
        assert dm.depth.shape == (10, 20)
        assert dm.confidence.shape == (10, 20)
        assert dm.width == 20
        assert dm.height == 10
        assert dm.latency_ms == 42.0
        assert dm.backend == "dummy"

    def test_shape_property(self) -> None:
        depth = np.zeros((5, 8), dtype=np.float32)
        conf = np.ones((5, 8), dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        assert dm.shape == (5, 8)

    def test_width_height_auto_inferred(self) -> None:
        depth = np.zeros((4, 7), dtype=np.float32)
        conf = np.ones((4, 7), dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        assert dm.width == 7
        assert dm.height == 4


class TestDepthMapValidation:
    def test_shape_mismatch_rejected(self) -> None:
        depth = np.zeros((10, 20), dtype=np.float32)
        conf = np.zeros((10, 30), dtype=np.float32)
        with pytest.raises(ValueError, match="depth shape .* != confidence shape"):
            DepthMap(depth=depth, confidence=conf)

    def test_3d_depth_rejected(self) -> None:
        depth = np.zeros((10, 20, 3), dtype=np.float32)
        conf = np.zeros((10, 20, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="depth must be 2D"):
            DepthMap(depth=depth, confidence=conf)

    def test_negative_latency_rejected(self) -> None:
        depth = np.zeros((4, 4), dtype=np.float32)
        conf = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="latency_ms must be >= 0"):
            DepthMap(depth=depth, confidence=conf, latency_ms=-1.0)

    def test_dtype_coerced_to_float32(self) -> None:
        depth = np.full((3, 3), 1.5, dtype=np.float64)
        conf = np.full((3, 3), 0.5, dtype=np.float64)
        dm = DepthMap(depth=depth, confidence=conf)
        assert dm.depth.dtype == np.float32
        assert dm.confidence.dtype == np.float32


class TestConfidenceClipping:
    def test_confidence_clipped_to_0_1(self) -> None:
        depth = np.zeros((2, 2), dtype=np.float32)
        conf = np.array([[1.5, -0.3], [0.5, 2.0]], dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        assert dm.confidence.min() >= 0.0
        assert dm.confidence.max() <= 1.0
        assert dm.confidence[0, 0] == 1.0
        assert dm.confidence[0, 1] == 0.0
        assert dm.confidence[1, 1] == 1.0


class TestSerialization:
    def test_to_dict(self) -> None:
        depth = np.array([[1.0, 2.0]], dtype=np.float32)
        conf = np.array([[0.9, 0.1]], dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf, backend="dummy", latency_ms=10.0)
        d = dm.to_dict()
        assert d["backend"] == "dummy"
        assert d["latency_ms"] == 10.0
        assert d["width"] == 2
        assert d["height"] == 1
        assert d["depth"] == [[1.0, 2.0]]
        assert d["confidence"][0] == pytest.approx([0.9, 0.1], abs=1e-5)

    def test_summary_valid_pixels(self) -> None:
        depth = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        conf = np.full((1, 3), 0.8, dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        s = dm.summary()
        assert s["valid_pixels"] == 2  # 两个非零
        assert s["total_pixels"] == 3
        assert s["depth_min"] == 1.0
        assert s["depth_max"] == 2.0
        assert s["depth_mean"] == pytest.approx(1.5)

    def test_summary_all_zero_depth(self) -> None:
        depth = np.zeros((2, 2), dtype=np.float32)
        conf = np.zeros((2, 2), dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        s = dm.summary()
        assert s["valid_pixels"] == 0
        assert s["depth_min"] == 0.0
