"""M011 Feature-based Visual Odometry: ORB 特征 + RANSAC.

用 OpenCV 的 ORB 特征匹配 + cv2.solvePnP (或 essential matrix + recoverPose)
估计帧间相机运动, 累积成轨迹.

特点:
- 需要 OpenCV (opencv-python-headless).
- 需要相机内参 K (M006 CameraCalibration).
- 单目尺度模糊: 帧间平移只有方向, 无绝对尺度. 用 scale_factor 乘到合理量级.
- 失败帧 (特征不足 / RANSAC 失败) 返回 None, 由基类统计.

限制:
- 单目 VO 会有尺度漂移 (M012 VIO 用 IMU 修正).
- 全景图 (equirectangular) 不能直接用 ORB (透视图特征), 需先抽视图.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from modules.logging import get_logger

from .base import VisualOdometry
from .types import Pose

try:
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_LOG = get_logger("modules.vo.feature")

# ORB 特征数.
_DEFAULT_N_FEATURES = 1000
# 匹配距离阈值 (汉明距离).
_DEFAULT_MATCH_RATIO = 0.7  # Lowe's ratio test
# RANSAC 阈值.
_DEFAULT_RANSAC_THRESHOLD = 3.0
# 最少匹配数.
_MIN_MATCHES = 20
# 单目尺度因子 (米, 启发式).
_DEFAULT_SCALE_FACTOR = 0.1
# 灰度图维度.
_NDIM_GRAYSCALE = 2
# 灰度图占位尺寸.
_PLACEHOLDER_GRAY_SIZE = (64, 64)


class FeatureVO(VisualOdometry):
    """ORB 特征 + RANSAC Visual Odometry.

    首帧为单位 pose. 后续每帧用 ORB 匹配 + cv2.recoverPose 估计相对运动.
    """

    backend_name = "orb"

    def __init__(
        self,
        camera_matrix: np.ndarray,
        device: str = "cpu",
        n_features: int = _DEFAULT_N_FEATURES,
        scale_factor: float = _DEFAULT_SCALE_FACTOR,
    ) -> None:
        super().__init__(device=device)
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                "opencv not installed; run: uv pip install opencv-python-headless "
                "or use DummyVisualOdometry for testing"
            )
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError(f"camera_matrix must be (3,3), got {self.K.shape}")
        self.n_features = n_features
        self.scale_factor = scale_factor
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _estimate_poses(self, frames: Sequence[Any]) -> list[Pose | None]:
        """帧序列 → 位姿序列."""
        if len(frames) == 0:
            return []

        poses: list[Pose | None] = []
        # 首帧: 单位 pose.
        first_frame = frames[0]
        poses.append(
            Pose.identity(
                timestamp=getattr(first_frame, "timestamp", 0.0),
                frame_id=getattr(first_frame, "frame_id", 0),
            )
        )

        # 累积位姿 (world_to_camera).
        cumulative = np.eye(4, dtype=np.float64)

        prev_gray = self._to_gray(first_frame)
        prev_kp, prev_des = self._orb.detectAndCompute(prev_gray, None)

        for i in range(1, len(frames)):
            frame = frames[i]
            gray = self._to_gray(frame)
            kp, des = self._orb.detectAndCompute(gray, None)

            pose = None
            if prev_des is not None and des is not None and len(prev_des) > 0 and len(des) > 0:
                matches = self._matcher.match(prev_des, des)
                # 距离排序, 取好的.
                matches = sorted(matches, key=lambda m: m.distance)
                if len(matches) >= _MIN_MATCHES:
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                    pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

                    E, mask = cv2.findEssentialMat(
                        pts_prev,
                        pts_curr,
                        self.K,
                        method=cv2.RANSAC,
                        threshold=_DEFAULT_RANSAC_THRESHOLD,
                    )
                    if E is not None and mask is not None:
                        inliers = int(mask.sum())
                        if inliers >= _MIN_MATCHES // 2:
                            _, R_rel, t_rel, _ = cv2.recoverPose(
                                E, pts_prev, pts_curr, self.K, mask=mask
                            )
                            # 相对运动 → 4x4.
                            T_rel = np.eye(4, dtype=np.float64)
                            T_rel[:3, :3] = R_rel
                            T_rel[:3, 3] = t_rel.ravel() * self.scale_factor
                            # 累积: T_world_curr = T_world_prev @ T_prev_curr
                            cumulative = cumulative @ T_rel
                            pose = Pose(
                                matrix=cumulative.copy(),
                                timestamp=getattr(frame, "timestamp", float(i)),
                                frame_id=getattr(frame, "frame_id", i),
                                confidence=inliers / len(matches),
                                metadata={
                                    "n_matches": len(matches),
                                    "n_inliers": inliers,
                                },
                            )

            poses.append(pose)
            prev_gray = gray
            prev_kp, prev_des = kp, des

        return poses

    def _to_gray(self, frame: Any) -> np.ndarray:
        """Frame → 灰度图."""
        image = getattr(frame, "image", frame)
        if image is None:
            # 占位黑图.
            return np.zeros(_PLACEHOLDER_GRAY_SIZE, dtype=np.uint8)
        img = np.asarray(image)
        if img.ndim != _NDIM_GRAYSCALE:  # 彩色 → 灰度
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            {
                "n_features": self.n_features,
                "scale_factor": self.scale_factor,
                "cv2_available": _CV2_AVAILABLE,
            }
        )
        return d
