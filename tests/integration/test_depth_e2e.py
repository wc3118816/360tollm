"""M009 集成测试: 深度估计端到端.

验证:
1. 从 Frame.image 到 DepthMap 的完整链路
2. 批量推理离线数据集帧
3. AC: 输出深度图 + 置信度 + 延迟
4. DepthMap → Frame.depth 注入
5. 多帧延迟统计
"""

from __future__ import annotations

import numpy as np

from modules.depth import DepthMap, DummyDepthEstimator, create_depth_estimator
from modules.ingest.types import Frame


def _make_frame(h: int = 64, w: int = 64, frame_id: int = 0) -> Frame:
    """生成带图像的测试 Frame."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.uint8)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    brightness = (255 * np.exp(-r / (h / 4))).astype(np.uint8)
    img = np.stack([brightness, brightness, brightness], axis=-1)
    return Frame(frame_id=frame_id, timestamp=float(frame_id), image=img, source="test")


class TestFrameToDepthMap:
    """Frame.image → DepthMap."""

    def test_estimate_from_frame(self) -> None:
        est = DummyDepthEstimator()
        frame = _make_frame(64, 64)
        dm = est.estimate(frame.image)
        assert isinstance(dm, DepthMap)
        assert dm.depth.shape == (64, 64)

    def test_inject_depth_into_frame(self) -> None:
        """DepthMap.depth 可直接赋给 Frame.depth."""
        est = DummyDepthEstimator()
        frame = _make_frame(32, 32)
        dm = est.estimate(frame.image)
        frame.depth = dm.depth
        assert frame.depth is not None
        assert frame.depth.shape == (32, 32)
        assert frame.depth.dtype == np.float32


class TestBatchOfflineData:
    """AC: 可对离线数据批量推理."""

    def test_batch_estimate_frames(self) -> None:
        est = DummyDepthEstimator()
        frames = [_make_frame(32, 32, i) for i in range(5)]
        results = est.estimate_batch([f.image for f in frames])
        assert len(results) == 5
        for i, dm in enumerate(results):
            assert dm.depth.shape == (32, 32)
            assert dm.latency_ms > 0
            assert dm.metadata["batch_index"] == i

    def test_batch_consistent_latency(self) -> None:
        """批量推理延迟统计稳定 (5 帧平均 latency 与单帧一致量级)."""
        est = DummyDepthEstimator()
        frames = [_make_frame(32, 32) for _ in range(5)]
        # 单帧基线.
        single = est.estimate(frames[0].image)
        single_lat = single.latency_ms
        # 批量.
        est2 = DummyDepthEstimator()
        results = est2.estimate_batch([f.image for f in frames])
        avg_batch = sum(dm.latency_ms for dm in results) / len(results)
        # 同量级 (允许 10x 容差, 首帧可能慢).
        assert avg_batch < single_lat * 10 or avg_batch < 10.0


class TestACCompliance:
    """M009 AC 三项要求."""

    def test_output_depth_and_confidence(self) -> None:
        """AC: 输出深度图与置信度."""
        est = DummyDepthEstimator()
        dm = est.estimate(_make_frame(32, 32).image)
        assert dm.depth is not None
        assert dm.confidence is not None
        assert dm.depth.dtype == np.float32
        assert dm.confidence.dtype == np.float32
        assert dm.depth.shape == dm.confidence.shape

    def test_latency_recorded(self) -> None:
        """AC: 记录推理延迟."""
        est = DummyDepthEstimator()
        dm = est.estimate(_make_frame(16, 16).image)
        assert dm.latency_ms > 0
        assert isinstance(dm.latency_ms, float)

    def test_batch_inference(self) -> None:
        """AC: 可对离线数据批量推理."""
        est = DummyDepthEstimator()
        imgs = [_make_frame(16, 16, i).image for i in range(10)]
        results = est.estimate_batch(imgs)
        assert len(results) == 10
        assert est.stats.n_frames == 10


class TestStatsPersistence:
    def test_multi_frame_stats(self) -> None:
        """多帧后 stats 累积正确."""
        est = DummyDepthEstimator()
        for i in range(20):
            est.estimate(_make_frame(16, 16, i).image)
        assert est.stats.n_frames == 20
        assert est.stats.avg_latency_ms > 0
        assert est.stats.p95_latency_ms > 0
        d = est.stats.as_dict()
        assert d["n_frames"] == 20
        assert "avg_latency_ms" in d
        assert "p95_latency_ms" in d


class TestFromSettings:
    def test_create_from_default_config(self) -> None:
        """工厂从 settings 创建 (backend=dummy)."""
        est = create_depth_estimator()
        dm = est.estimate(_make_frame(32, 32).image)
        assert dm.backend in ("dummy", "midas")
        assert dm.depth.shape[0] == 32
