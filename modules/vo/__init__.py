"""M011 Visual Odometry 子包.

公开 API:
    Pose                - 单个位姿 (SE3 4x4 + timestamp + confidence)
    Trajectory          - 有序 Pose 序列 (轨迹)
    VisualOdometry      - VO 抽象基类 (estimate(frames) -> Trajectory)
    VOStats             - 运行统计 (n_frames / failure_rate / avg_latency)
    DummyVisualOdometry - 合成基线 (无 ML 依赖)
    FeatureVO           - ORB 特征 + RANSAC (需 OpenCV + 内参)
    create_vo           - 工厂函数 (从 SLAMCfg 创建)

用法:
    from modules.vo import create_vo
    vo = create_vo()  # 从 settings 读 cfg
    traj = vo.estimate(frames)
    print(traj.summary())  # {n_poses, has_nan, total_displacement, ...}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import VisualOdometry, VOStats
from .dummy_vo import DummyVisualOdometry
from .feature_vo import _CV2_AVAILABLE, FeatureVO
from .types import Pose, Trajectory

if TYPE_CHECKING:
    import numpy as np

    from modules.config.settings import DepthCfg as _DepthCfg  # noqa: F401
    from modules.config.settings import SLAMCfg


def create_vo(
    cfg: SLAMCfg | None = None,
    camera_matrix: np.ndarray | None = None,
) -> VisualOdometry:
    """从 SLAMCfg 创建 Visual Odometry 估计器.

    Args:
        cfg: SLAMCfg. None 时从 settings 读取.
        camera_matrix: 相机内参 K (3x3). FeatureVO 必需; Dummy 忽略.

    Returns:
        VisualOdometry 实例.

    Raises:
        ValueError: 未知 backend.
        RuntimeError: backend 需要但 OpenCV 未安装.
    """
    if cfg is None:
        from modules.config import get_settings

        cfg = get_settings().slam

    backend = cfg.backend.lower()
    if backend in ("none", "dummy"):
        return DummyVisualOdometry()
    if backend in ("orb", "feature", "feature_vo"):
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                f"backend '{backend}' requires opencv; "
                "run: uv pip install opencv-python-headless "
                "or set slam.backend=dummy"
            )
        if camera_matrix is None:
            raise ValueError(
                f"backend '{backend}' requires camera_matrix (3x3); "
                "pass it explicitly or use slam.backend=dummy"
            )
        return FeatureVO(camera_matrix=camera_matrix)
    raise ValueError(
        f"unknown VO backend: '{cfg.backend}'. supported: none | dummy | orb | feature | feature_vo"
    )


__all__ = [
    "Pose",
    "Trajectory",
    "VisualOdometry",
    "VOStats",
    "DummyVisualOdometry",
    "FeatureVO",
    "create_vo",
]
