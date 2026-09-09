"""M006 棋盘格标定器: OpenCV calibrateCamera 封装.

流程:
1. 从图像序列检测棋盘格角点 (cv2.findChessboardCorners).
2. 亚像素精化 (cv2.cornerSubPix).
3. 调用 cv2.calibrateCamera 计算 K + D + rvecs + tvecs.
4. 计算重投影误差 (AC 要求).
5. 误差 > cfg.max_reproj_error 时抛 CalibrationError.

用法:
    from modules.camera import Calibrator
    from modules.config import get_settings
    cal = Calibrator(get_settings().calibration)
    result = cal.calibrate(image_paths, source="chessboard_9x6")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from modules.camera.calibration import CameraCalibration
from modules.config.settings import CalibrationCfg
from modules.logging import get_logger

try:
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_COLOR_NDIM = 3  # 彩色图维度


class CalibrationError(Exception):
    """标定失败异常."""


class Calibrator:
    """棋盘格相机标定器.

    AC:
    - 提供重投影误差 → reproj_error 字段
    - 误差阈值可配置 → cfg.max_reproj_error
    - 标定参数可被运行时加载 → 通过 save/load_calibration (io.py)
    """

    def __init__(self, cfg: CalibrationCfg | None = None) -> None:
        self._cfg = cfg or _default_calibration_cfg()
        self._log = get_logger("modules.camera.calibrator")
        if not _CV2_AVAILABLE:
            self._log.warning("opencv not installed; Calibrator will be unusable")

        # 棋盘格内角点尺寸 (cols, rows).
        self._pattern_size = (self._cfg.chessboard_cols, self._cfg.chessboard_rows)
        self._square_size = self._cfg.square_size

        # 预计算 3D 世界坐标 (Z=0 平面). OpenCV 要求 float32.
        self._obj_points = np.zeros(
            (self._cfg.chessboard_cols * self._cfg.chessboard_rows, 3),
            dtype=np.float32,
        )
        self._obj_points[:, :2] = np.mgrid[
            0 : self._cfg.chessboard_cols, 0 : self._cfg.chessboard_rows
        ].T.reshape(-1, 2)
        self._obj_points *= np.float32(self._square_size)

    def calibrate(
        self,
        image_paths: list[str | Path] | list[np.ndarray],
        source: str = "chessboard",
    ) -> CameraCalibration:
        """从图像序列标定相机.

        Args:
            image_paths: 图像路径列表 或 已加载的 ndarray 列表.
            source: 标定来源标签 (写入结果的 metadata).

        Returns:
            CameraCalibration 结果.

        Raises:
            CalibrationError: 有效图像不足 / 重投影误差超阈值.
        """
        if not _CV2_AVAILABLE:
            raise CalibrationError("opencv not installed; cannot calibrate")

        # 灰度图 + 角点检测.
        obj_points: list[np.ndarray] = []  # 3D 点 (每图一份)
        img_points: list[np.ndarray] = []  # 2D 角点 (每图一份)
        image_size: tuple[int, int] | None = None  # (width, height)

        for item in image_paths:
            gray = self._load_gray(item)
            if gray is None:
                continue
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(
                gray,
                self._pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                self._log.debug("chessboard not found", item=self._item_name(item))
                continue

            # 亚像素精化.
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            obj_points.append(self._obj_points.copy())
            img_points.append(corners_refined)

        n_valid = len(obj_points)
        if n_valid < self._cfg.min_valid_images:
            raise CalibrationError(
                f"insufficient valid images: {n_valid} < {self._cfg.min_valid_images}"
            )
        assert image_size is not None

        # 标定.
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        if not ret:
            raise CalibrationError("cv2.calibrateCamera returned failure")

        # 重投影误差 (AC 要求).
        reproj_error = self._compute_reproj_error(obj_points, img_points, rvecs, tvecs, K, dist)

        self._log.info(
            "calibration done",
            n_images=n_valid,
            reproj_error=reproj_error,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )

        if reproj_error > self._cfg.max_reproj_error:
            raise CalibrationError(
                f"reprojection error {reproj_error:.4f}px exceeds threshold "
                f"{self._cfg.max_reproj_error}px"
            )

        return CameraCalibration(
            camera_matrix=K,
            dist_coeffs=dist.ravel(),
            reproj_error=reproj_error,
            width=image_size[0],
            height=image_size[1],
            n_images=n_valid,
            n_corners=self._pattern_size,
            square_size=self._square_size,
            calibrated_at=datetime.now(UTC).isoformat(),
            source=source,
        )

    def _load_gray(self, item: str | Path | np.ndarray) -> np.ndarray | None:
        """加载灰度图. 返回 None 表示加载失败."""
        try:
            if isinstance(item, np.ndarray):
                img = item
            else:
                import cv2  # type: ignore

                img = cv2.imread(str(item))
                if img is None:
                    return None
            if img.ndim == _COLOR_NDIM:
                import cv2  # type: ignore

                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except Exception as e:  # noqa: BLE001
            self._log.debug("image load failed", item=self._item_name(item), error=str(e))
            return None

    @staticmethod
    def _item_name(item: Any) -> str:
        if isinstance(item, (str, Path)):
            return str(item)
        return f"<ndarray shape={getattr(item, 'shape', '?')}>"

    @staticmethod
    def _compute_reproj_error(  # noqa: PLR0917
        obj_points: list[np.ndarray],
        img_points: list[np.ndarray],
        rvecs: list[np.ndarray],
        tvecs: list[np.ndarray],
        K: np.ndarray,
        dist: np.ndarray,
    ) -> float:
        """计算平均重投影误差 (像素). 用 numpy 避免 cv2.norm 类型不匹配."""
        import cv2  # type: ignore

        total_error = 0.0
        total_images = 0
        for i, (obj, img) in enumerate(zip(obj_points, img_points, strict=False)):
            projected, _ = cv2.projectPoints(
                np.ascontiguousarray(obj, dtype=np.float32),
                rvecs[i],
                tvecs[i],
                K,
                dist,
            )
            # img / projected 都是 (N,1,2) float32; reshape 到 (N,2).
            img_flat = img.reshape(-1, 2).astype(np.float64)
            proj_flat = projected.reshape(-1, 2).astype(np.float64)
            # 每点欧氏距离, 取平均.
            dists = np.linalg.norm(img_flat - proj_flat, axis=1)
            total_error += float(np.mean(dists))
            total_images += 1
        return total_error / total_images if total_images > 0 else 0.0


def _default_calibration_cfg() -> CalibrationCfg:
    from modules.config import get_settings

    return get_settings().calibration
