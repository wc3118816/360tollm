"""M019 目标检测: Dummy 检测器 (无 ML 依赖, 合成基线).

DummyDetector:
- 不加载任何模型, 纯 numpy 合成检测.
- 在图像随机位置生成 N 个 bbox + label + confidence.
- 模拟推理延迟 (sleep), 用于验证 AC (设定 FPS / 延迟可测).
- label 从默认类别列表中随机选择.

用途:
- MVP 流程验证 (M015→M021→M026 依赖检测输出).
- 单测/集成测试的 mock.
- 延迟/统计模块的验证.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.logging import get_logger

from .base import Detector
from .types import Detection

_LOG = get_logger("modules.detection.dummy")

# 默认类别列表 (COCO 风格子集).
_DEFAULT_LABELS = [
    "person",
    "chair",
    "cup",
    "bottle",
    "laptop",
    "book",
    "table",
    "door",
    "tv",
    "plant",
]
# 默认每帧检测数.
_DEFAULT_N_DETECTIONS = 3
# 默认模拟延迟 (毫秒).
_DEFAULT_LATENCY_MS = 5.0
# 默认置信度范围.
_DEFAULT_CONF_MIN = 0.5
_DEFAULT_CONF_MAX = 0.99
# bbox 最小尺寸.
_MIN_BBOX_SIZE = 10


@dataclass
class DummyDetectorConfig:
    """Dummy 检测器配置."""

    n_detections: int = _DEFAULT_N_DETECTIONS  # 每帧检测数
    latency_ms: float = _DEFAULT_LATENCY_MS  # 模拟延迟 (毫秒)
    labels: list[str] | None = None  # 类别列表 (None 用默认)
    conf_min: float = _DEFAULT_CONF_MIN
    conf_max: float = _DEFAULT_CONF_MAX
    seed: int | None = 42  # 随机种子 (None = 不固定)


class DummyDetector(Detector):
    """合成目标检测器 (无 ML 依赖).

    用法:
        det = DummyDetector()
        dl = det.detect(image)
        print(dl.summary())
    """

    backend_name = "dummy"

    def __init__(
        self,
        cfg: DummyDetectorConfig | None = None,
        device: str = "cpu",
        max_resolution: int = 1024,
    ) -> None:
        super().__init__(device=device, max_resolution=max_resolution)
        self.cfg = cfg or DummyDetectorConfig()
        self._labels = self.cfg.labels or _DEFAULT_LABELS
        self._rng = (
            np.random.RandomState(self.cfg.seed)
            if self.cfg.seed is not None
            else np.random.RandomState()
        )
        self._log = _LOG.bind(backend=self.backend_name)

    def _detect(self, image: np.ndarray) -> list[Detection]:
        """合成 N 个随机 detection."""
        h, w = image.shape[:2]
        n = self.cfg.n_detections
        detections: list[Detection] = []

        for i in range(n):
            # 随机 bbox (确保在图像内 + 最小尺寸).
            bw = self._rng.randint(_MIN_BBOX_SIZE, max(_MIN_BBOX_SIZE + 1, w // 2))
            bh = self._rng.randint(_MIN_BBOX_SIZE, max(_MIN_BBOX_SIZE + 1, h // 2))
            x_min = int(self._rng.randint(0, max(1, w - bw)))
            y_min = int(self._rng.randint(0, max(1, h - bh)))
            x_max = x_min + bw
            y_max = y_min + bh

            # 随机类别 + 置信度.
            label = str(self._labels[self._rng.randint(0, len(self._labels))])
            label_id = int(self._labels.index(label))
            conf = float(self._rng.uniform(self.cfg.conf_min, self.cfg.conf_max))

            det = Detection(
                bbox=np.array([x_min, y_min, x_max, y_max], dtype=np.int32),
                label=label,
                label_id=label_id,
                confidence=conf,
                metadata={"dummy_index": i},
            )
            detections.append(det)

        # 模拟推理延迟.
        if self.cfg.latency_ms > 0:
            time.sleep(self.cfg.latency_ms / 1000.0)

        return detections

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "device": self.device,
            "n_detections": self.cfg.n_detections,
            "latency_ms": self.cfg.latency_ms,
            "labels": self._labels,
            "seed": self.cfg.seed,
        }
