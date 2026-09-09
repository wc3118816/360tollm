"""M009 深度估计子包.

公开 API:
    DepthMap               - 深度结果数据结构 (depth + confidence + latency)
    DepthEstimator         - 估计器 ABC (estimate / estimate_batch)
    DepthStats             - 运行统计 (n_frames / avg_latency / p95)
    DummyDepthEstimator    - 合成基线 (无 ML 依赖)
    MiDaSDepthEstimator    - MiDaS 真实基线 (需 torch)
    create_depth_estimator - 工厂函数 (从 DepthCfg 创建)

用法:
    from modules.depth import create_depth_estimator
    est = create_depth_estimator()  # 从 settings 读 cfg
    dm = est.estimate(image)
    print(dm.summary())  # {backend, latency_ms, depth_min, ...}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DepthEstimator, DepthStats
from .dummy_estimator import DummyDepthEstimator
from .midas_estimator import _TORCH_AVAILABLE, MiDaSDepthEstimator
from .types import DepthMap

if TYPE_CHECKING:
    from modules.config.settings import DepthCfg


def create_depth_estimator(
    cfg: DepthCfg | None = None,
) -> DepthEstimator:
    """从 DepthCfg 创建深度估计器.

    Args:
        cfg: DepthCfg. None 时从 settings 读取.

    Returns:
        DepthEstimator 实例.

    Raises:
        ValueError: 未知 backend.
        RuntimeError: backend 需要但 torch 未安装.
    """
    if cfg is None:
        from modules.config import get_settings

        cfg = get_settings().depth

    backend = cfg.backend.lower()
    if backend in ("none", "dummy"):
        return DummyDepthEstimator(device=cfg.device, max_resolution=cfg.max_resolution)
    if backend in ("midas", "midas_small", "midas_large"):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                f"backend '{backend}' requires torch; "
                "run: uv pip install torch torchvision "
                "or set depth.backend=dummy"
            )
        model_type = "MiDaS_small" if backend == "midas_small" else "DPT_Large"
        return MiDaSDepthEstimator(
            device=cfg.device,
            max_resolution=cfg.max_resolution,
            model_type=model_type,
        )
    raise ValueError(
        f"unknown depth backend: '{cfg.backend}'. "
        f"supported: none | dummy | midas | midas_small | midas_large"
    )


__all__ = [
    "DepthMap",
    "DepthEstimator",
    "DepthStats",
    "DummyDepthEstimator",
    "MiDaSDepthEstimator",
    "create_depth_estimator",
]
