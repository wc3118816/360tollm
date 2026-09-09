"""M009 单元测试: DepthEstimator + DummyDepthEstimator + 工厂.

覆盖:
- Dummy 估计器: 输出形状/范围/置信度/确定性/latency 记录
- 批量推理: 长度一致 + 每帧 latency 记录
- 预处理: 灰度转 RGB + 下采样
- 工厂: 从 cfg 创建 + 未知 backend 报错 + torch 未装时 MiDaS 报错
- 统计: avg / p95
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.config.settings import DepthCfg
from modules.depth import (
    DepthEstimator,
    DepthMap,
    DepthStats,
    DummyDepthEstimator,
    create_depth_estimator,
)


def _make_image(h: int = 64, w: int = 64) -> np.ndarray:
    """生成测试 RGB 图像 (中心亮, 边缘暗)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.uint8)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.sqrt(cy**2 + cx**2)
    brightness = (255 * (1 - r / r_max)).astype(np.uint8)
    return np.stack([brightness, brightness, brightness], axis=-1)


class TestDummyDepthEstimator:
    def test_estimate_output_shape(self) -> None:
        est = DummyDepthEstimator()
        img = _make_image(64, 64)
        dm = est.estimate(img)
        assert isinstance(dm, DepthMap)
        assert dm.depth.shape == (64, 64)
        assert dm.confidence.shape == (64, 64)
        assert dm.width == 64
        assert dm.height == 64

    def test_estimate_depth_range(self) -> None:
        est = DummyDepthEstimator(near=1.0, far=5.0)
        img = _make_image(32, 32)
        dm = est.estimate(img)
        assert dm.depth.min() >= 1.0
        assert dm.depth.max() <= 5.0

    def test_estimate_confidence_range(self) -> None:
        est = DummyDepthEstimator()
        dm = est.estimate(_make_image(16, 16))
        assert dm.confidence.min() >= 0.0
        assert dm.confidence.max() <= 1.0

    def test_center_closer_than_edges(self) -> None:
        """中心深度 < 边缘深度 (中心近)."""
        est = DummyDepthEstimator()
        img = _make_image(64, 64)
        dm = est.estimate(img)
        center_depth = dm.depth[32, 32]
        corner_depth = dm.depth[0, 0]
        assert center_depth < corner_depth

    def test_deterministic_output(self) -> None:
        """相同输入 → 相同输出."""
        est = DummyDepthEstimator()
        img = _make_image(32, 32)
        dm1 = est.estimate(img)
        dm2 = est.estimate(img)
        assert np.array_equal(dm1.depth, dm2.depth)

    def test_latency_recorded(self) -> None:
        """AC: 记录推理延迟. latency_ms > 0."""
        est = DummyDepthEstimator()
        dm = est.estimate(_make_image(32, 32))
        assert dm.latency_ms > 0
        assert dm.backend == "dummy"
        assert "device" in dm.metadata
        assert "max_resolution" in dm.metadata

    def test_metadata_contains_near_far(self) -> None:
        est = DummyDepthEstimator(near=0.5, far=10.0)
        dm = est.estimate(_make_image(8, 8))
        assert dm.metadata["near"] == 0.5
        assert dm.metadata["far"] == 10.0
        assert dm.metadata["synthetic"] is True

    def test_describe(self) -> None:
        est = DummyDepthEstimator(device="cpu", max_resolution=512)
        d = est.describe()
        assert d["backend"] == "dummy"
        assert d["device"] == "cpu"
        assert d["max_resolution"] == 512
        assert d["near"] == 0.5
        assert d["far"] == 10.0


class TestDepthEstimatorPreprocess:
    def test_grayscale_input_accepted(self) -> None:
        """灰度输入 (H, W) 自动转 RGB."""
        est = DummyDepthEstimator()
        gray = np.full((16, 16), 128, dtype=np.uint8)
        dm = est.estimate(gray)
        assert dm.depth.shape == (16, 16)

    def test_invalid_shape_rejected(self) -> None:
        est = DummyDepthEstimator()
        bad = np.zeros((16, 16, 4), dtype=np.uint8)  # 4 通道
        with pytest.raises(ValueError, match="image must be"):
            est.estimate(bad)

    def test_downsampling(self) -> None:
        """max_resolution=32 时, 64x64 输入下采样到 ≤32 边长."""
        est = DummyDepthEstimator(max_resolution=32)
        img = _make_image(64, 64)
        dm = est.estimate(img)
        # 输出深度图的尺寸 ≤ 32 (下采样后)
        assert max(dm.depth.shape) <= 32

    def test_small_image_not_upscaled(self) -> None:
        """小图不放大."""
        est = DummyDepthEstimator(max_resolution=1024)
        img = _make_image(16, 16)
        dm = est.estimate(img)
        assert dm.depth.shape == (16, 16)


class TestBatchInference:
    """AC: 可对离线数据批量推理."""

    def test_batch_returns_list(self) -> None:
        est = DummyDepthEstimator()
        imgs = [_make_image(32, 32) for _ in range(5)]
        results = est.estimate_batch(imgs)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_batch_each_has_depth(self) -> None:
        est = DummyDepthEstimator()
        imgs = [_make_image(16, 16) for _ in range(3)]
        results = est.estimate_batch(imgs)
        for dm in results:
            assert isinstance(dm, DepthMap)
            assert dm.depth.shape == (16, 16)

    def test_batch_latency_recorded(self) -> None:
        est = DummyDepthEstimator()
        imgs = [_make_image(16, 16) for _ in range(3)]
        results = est.estimate_batch(imgs)
        for dm in results:
            assert dm.latency_ms > 0
            assert "batch_index" in dm.metadata

    def test_batch_stats_accumulated(self) -> None:
        est = DummyDepthEstimator()
        imgs = [_make_image(16, 16) for _ in range(10)]
        est.estimate_batch(imgs)
        assert est.stats.n_frames == 10
        assert est.stats.avg_latency_ms > 0


class TestDepthStats:
    def test_empty_stats(self) -> None:
        s = DepthStats()
        assert s.n_frames == 0
        assert s.avg_latency_ms == 0.0
        assert s.p95_latency_ms == 0.0

    def test_avg_latency(self) -> None:
        s = DepthStats()
        for lat in [10.0, 20.0, 30.0]:
            s.record(lat)
        assert s.avg_latency_ms == pytest.approx(20.0)

    def test_p95_latency(self) -> None:
        s = DepthStats()
        for i in range(100):
            s.record(float(i))  # 0..99
        # 95th percentile: ~95
        assert s.p95_latency_ms >= 90.0
        assert s.p95_latency_ms <= 99.0

    def test_latency_list_capped(self) -> None:
        """超过 1000 条时自动截断."""
        s = DepthStats()
        for _ in range(1500):
            s.record(1.0)
        assert len(s.latencies_ms) <= 1000


class TestFactory:
    def test_create_dummy_from_cfg(self) -> None:
        cfg = DepthCfg(backend="dummy", device="cpu", max_resolution=512)
        est = create_depth_estimator(cfg)
        assert isinstance(est, DummyDepthEstimator)
        assert est.max_resolution == 512

    def test_create_dummy_from_none_backend(self) -> None:
        cfg = DepthCfg(backend="none")
        est = create_depth_estimator(cfg)
        assert isinstance(est, DummyDepthEstimator)

    def test_unknown_backend_raises(self) -> None:
        cfg = DepthCfg(backend="nonexistent")
        with pytest.raises(ValueError, match="unknown depth backend"):
            create_depth_estimator(cfg)

    def test_create_from_settings(self) -> None:
        """None 时从 settings 读取."""
        est = create_depth_estimator()
        assert isinstance(est, DepthEstimator)

    def test_midas_without_torch_raises(self) -> None:
        """MiDaS backend 但 torch 未安装 → RuntimeError."""
        from modules.depth.midas_estimator import _TORCH_AVAILABLE

        if _TORCH_AVAILABLE:
            pytest.skip("torch installed, cannot test RuntimeError path")
        cfg = DepthCfg(backend="midas")
        with pytest.raises(RuntimeError, match="requires torch"):
            create_depth_estimator(cfg)
