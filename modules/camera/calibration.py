"""M006 CameraCalibration: 标定结果数据结构.

承载:
- 内参矩阵 K (3x3): fx, fy, cx, cy
- 畸变系数 D (k1, k2, p1, p2, k3): OpenCV 标准 5 参数模型
- 重投影误差 reproj_error (像素): AC 要求
- 图像尺寸 width/height
- 标定时间/来源等元数据

与 M005 intrinsics.json 格式兼容: 可序列化为 JSON, 也可从 JSON 反序列化.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_MIN_DIST_COEFFS = 5  # OpenCV 至少需要 5 个畸变参数 (k1,k2,p1,p2,k3)


@dataclass(slots=True)
class CameraCalibration:
    """相机标定结果.

    Attributes:
        camera_matrix: 3x3 内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]].
        dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3] (OpenCV 5 参数).
        reproj_error: 平均重投影误差 (像素). 越小越好, 通常 < 0.5px 为优.
        width: 图像宽度 (像素).
        height: 图像高度 (像素).
        n_images: 用于标定的图像数.
        n_corners: 每幅图检测到的角点数 (棋盘格 inner corners).
        square_size: 棋盘格格子边长 (米, 物理尺寸).
        calibrated_at: 标定时间戳 (ISO 格式字符串).
        source: 标定来源 (如 "chessboard_9x6_30imgs" / "realsense_factory").
    """

    camera_matrix: np.ndarray  # (3,3) float64
    dist_coeffs: np.ndarray  # (5,) float64
    reproj_error: float
    width: int
    height: int
    n_images: int = 0
    n_corners: tuple[int, int] = (0, 0)
    square_size: float = 0.0
    calibrated_at: str = ""
    source: str = "unknown"

    def __post_init__(self) -> None:
        # 类型与形状校验.
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).ravel()
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must be (3,3), got {self.camera_matrix.shape}")
        if self.dist_coeffs.size < _MIN_DIST_COEFFS:
            # OpenCV 至少需要 5 个畸变参数; 不足补零.
            padded = np.zeros(5, dtype=np.float64)
            padded[: self.dist_coeffs.size] = self.dist_coeffs
            self.dist_coeffs = padded
        if self.reproj_error < 0:
            raise ValueError(f"reproj_error must be >= 0, got {self.reproj_error}")

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容的 dict."""
        return {
            "type": "perspective",
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reproj_error": self.reproj_error,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "n_images": self.n_images,
            "n_corners": list(self.n_corners),
            "square_size": self.square_size,
            "calibrated_at": self.calibrated_at,
            "source": self.source,
        }

    def to_frame_intrinsics(self) -> dict[str, Any]:
        """转成 Frame.intrinsics 兼容格式 (与 M005 intrinsics.json 一致)."""
        return {
            "type": "perspective",
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "distortion": self.dist_coeffs.tolist(),
            "reproj_error": self.reproj_error,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraCalibration:
        """从 dict 反序列化."""
        return cls(
            camera_matrix=np.array(d.get("camera_matrix"), dtype=np.float64),
            dist_coeffs=np.array(d.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64),
            reproj_error=float(d.get("reproj_error", 0.0)),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            n_images=int(d.get("n_images", 0)),
            n_corners=tuple(d.get("n_corners", [0, 0])),
            square_size=float(d.get("square_size", 0.0)),
            calibrated_at=str(d.get("calibrated_at", "")),
            source=str(d.get("source", "unknown")),
        )
