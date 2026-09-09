"""相机标定子包 (M006).

公开 API:
    CameraCalibration   - 标定结果数据结构 (内参 K + 畸变 + 重投影误差)
    Calibrator          - 棋盘格标定器 (OpenCV calibrateCamera 封装)
    load_calibration    - 从 JSON 加载标定参数
    save_calibration    - 保存标定参数到 JSON
"""

from __future__ import annotations

from .calibration import CameraCalibration
from .calibrator import Calibrator
from .io import load_calibration, save_calibration

__all__ = [
    "CameraCalibration",
    "Calibrator",
    "load_calibration",
    "save_calibration",
]
