"""M009 Dummy 深度估计器: 合成基线, 无需 ML 依赖.

用图像亮度 + 中心先验生成"伪深度":
- 中心近 (深度小), 边缘远 (深度大) — 模拟针孔相机的径向距离衰减.
- 亮度高 → 近 (深度小), 暗区域 → 远 — 模拟"亮 = 近光源"启发式.
- 置信度: 中心高, 边缘低.

用途:
1. 无 torch / 无网络环境下的流水线测试.
2. M015 点云生成 / M021 2D→3D 的占位深度源.
3. 验证 DepthEstimator ABC + 批量推理 + 统计的完整链路.

注意: 输出是合成深度, 不反映真实几何. 真实场景请用 MiDaS / Metric3D / Zoe.
"""

from __future__ import annotations

import numpy as np

from .base import DepthEstimator
from .types import DepthMap

# 默认深度范围 (米).
_DEFAULT_NEAR = 0.5
_DEFAULT_FAR = 10.0


class DummyDepthEstimator(DepthEstimator):
    """合成深度估计器.

    生成确定性伪深度 (中心近 + 亮度启发 + 径向衰减).
    相同输入 → 相同输出 (便于测试).
    """

    backend_name = "dummy"

    def __init__(
        self,
        device: str = "cpu",
        max_resolution: int = 1024,
        near: float = _DEFAULT_NEAR,
        far: float = _DEFAULT_FAR,
    ) -> None:
        super().__init__(device=device, max_resolution=max_resolution)
        self.near = near
        self.far = far

    def _estimate(self, image: np.ndarray) -> DepthMap:
        """生成合成深度图.

        深度 = near + (far - near) * normalized_distance
        其中 normalized_distance ∈ [0, 1] 综合三个因素:
        1. 径向距离 (中心 0, 边缘 1)
        2. 归一化亮度 (亮 0, 暗 1)
        3. 两者加权 0.5/0.5
        """
        h, w = image.shape[:2]
        gray = np.mean(image, axis=2).astype(np.float32)  # (H, W)

        # 径向距离: 中心 0, 角点 1.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        r_norm = np.clip(r, 0.0, 1.0)

        # 亮度归一化 (0..255 → 0..1, 反转: 亮=近=0).
        b_norm = 1.0 - gray / 255.0

        # 综合.
        dist = 0.5 * r_norm + 0.5 * b_norm
        dist = np.clip(dist, 0.0, 1.0)
        depth = self.near + (self.far - self.near) * dist

        # 置信度: 中心高, 边缘低.
        confidence = 1.0 - r_norm * 0.8

        return DepthMap(
            depth=depth.astype(np.float32),
            confidence=confidence.astype(np.float32),
            backend=self.backend_name,
            metadata={"near": self.near, "far": self.far, "synthetic": True},
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update({"near": self.near, "far": self.far, "synthetic": True})
        return d
