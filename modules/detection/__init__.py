"""M019 目标检测子包.

公开 API:
    Detection             - 单个检测结果 (bbox + label + confidence)
    DetectionList         - 单帧检测结果集合 (含 latency + summary)
    Detector              - 检测器 ABC (detect / detect_batch)
    DetectionStats        - 运行统计 (n_frames / avg_latency / p95 / fps)
    DummyDetector         - 合成基线 (无 ML 依赖)
    DummyDetectorConfig   - Dummy 配置
    create_detector       - 工厂函数 (从 cfg 创建)

用法:
    from modules.detection import DummyDetector
    det = DummyDetector()
    dl = det.detect(image)
    print(dl.summary())  # {n_detections, latency_ms, fps, ...}
"""

from __future__ import annotations

from .base import DetectionStats, Detector
from .dummy_detector import DummyDetector, DummyDetectorConfig
from .types import Detection, DetectionList

__all__ = [
    "Detection",
    "DetectionList",
    "Detector",
    "DetectionStats",
    "DummyDetector",
    "DummyDetectorConfig",
    "create_detector",
]


def create_detector(backend: str = "dummy", **kwargs: object) -> Detector:
    """工厂函数: 从后端名创建检测器.

    Args:
        backend: "dummy" (目前唯一支持).
        **kwargs: 传给检测器构造函数.

    Returns:
        Detector.
    """
    backend_lower = backend.lower()
    if backend_lower == "dummy":
        return DummyDetector(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown backend: {backend}")
