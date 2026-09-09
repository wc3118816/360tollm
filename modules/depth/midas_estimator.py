"""M009 MiDaS 深度估计器: 真实基线 (相对深度).

MiDaS (https://github.com/isl-org/MiDaS) 是 Intel 的单目深度估计模型.
输出是相对深度 (越近越大), 需反转并归一化到 [near, far] 米范围.

特点:
- 相对深度 (非绝对米数), 需后处理映射到米.
- 支持 DPT-Large / MiDaS-small 等多个变体.
- 通过 torch.hub 下载, 首次加载需网络.

本实现:
- lazy import torch (未安装 torch 时优雅降级).
- model_type 可配 (DPT_Large / MiDaS_small).
- 输出归一化到 [near, far] 米 (线性映射).
- 置信度: 暂用常数 1.0 (MiDaS 不输出置信度, 真实置信度需 M010 评测).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from modules.logging import get_logger

from .base import DepthEstimator
from .types import DepthMap

_LOG = get_logger("modules.depth.midas")

# torch 可用性检测 (不强制依赖).
try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# 默认深度映射范围 (米). MiDaS 是相对深度, 需映射到米.
_DEFAULT_NEAR = 0.5
_DEFAULT_FAR = 20.0
_DEFAULT_MODEL = "DPT_Large"
# 归一化深度范围差阈值 (防止除零).
_DEPTH_RANGE_EPS = 1e-6


class MiDaSDepthEstimator(DepthEstimator):
    """MiDaS 单目深度估计器.

    首次调用 estimate() 时通过 torch.hub 加载模型.
    未安装 torch 时抛 RuntimeError (而非 ImportError, 因为是运行时依赖).
    """

    backend_name = "midas"

    def __init__(
        self,
        device: str = "cpu",
        max_resolution: int = 1024,
        model_type: str = _DEFAULT_MODEL,
        near: float = _DEFAULT_NEAR,
        far: float = _DEFAULT_FAR,
    ) -> None:
        super().__init__(device=device, max_resolution=max_resolution)
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch not installed; run: uv pip install torch torchvision "
                "or use DummyDepthEstimator for testing"
            )
        self.model_type = model_type
        self.near = near
        self.far = far
        self._model: Any = None
        self._transform: Any = None

    def warmup(self) -> None:
        """预热: 预先加载模型."""
        self._load_model()

    def _load_model(self) -> None:
        """通过 torch.hub 加载 MiDaS 模型."""
        if self._model is not None:
            return
        _LOG.info("loading MiDaS model", model_type=self.model_type)
        self._model = torch.hub.load("isl-org/MiDaS", self.model_type)
        self._model.to(self.device).eval()
        midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
        if self.model_type in {"DPT_Large", "DPT_Hybrid"}:
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform
        _LOG.info("MiDaS model loaded", device=self.device)

    def _estimate(self, image: np.ndarray) -> DepthMap:
        """MiDaS 推理 + 相对深度 → 米映射."""
        self._load_model()
        # 预处理: numpy → tensor.
        import cv2  # type: ignore

        # OpenCV 风格输入 (BGR); 本模块输入是 RGB, 先转 BGR.
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_batch = self._transform(img_bgr).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            # 插值到原分辨率.
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_rel = prediction.cpu().numpy().astype(np.float32)
        # MiDaS 输出: 越近值越大. 归一化到 [0, 1] (1=近, 0=远).
        d_min, d_max = float(depth_rel.min()), float(depth_rel.max())
        if d_max - d_min < _DEPTH_RANGE_EPS:
            depth_norm = np.zeros_like(depth_rel)
        else:
            depth_norm = (depth_rel - d_min) / (d_max - d_min)
        # 反转 (1=近→0米, 0=远→far米) 并映射到 [near, far].
        depth_m = self.near + (self.far - self.near) * (1.0 - depth_norm)
        # 置信度: MiDaS 不输出, 用常数.
        confidence = np.ones_like(depth_m, dtype=np.float32)
        return DepthMap(
            depth=depth_m.astype(np.float32),
            confidence=confidence,
            backend=self.backend_name,
            metadata={
                "model_type": self.model_type,
                "near": self.near,
                "far": self.far,
                "relative_depth": True,
            },
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            {
                "model_type": self.model_type,
                "near": self.near,
                "far": self.far,
                "relative_depth": True,
            }
        )
        return d
