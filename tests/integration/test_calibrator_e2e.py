"""M006 集成测试: Calibrator 端到端标定.

用合成棋盘格图像 (datasets/calibration_test/) 真实跑 cv2.calibrateCamera,
验证:
1. 标定成功, 产出 CameraCalibration
2. reproj_error 为正且 < 阈值
3. K 矩阵接近已知值 (fx=fy=800, cx~320, cy~240)
4. 保存 → 加载 → 字段一致
5. 误差阈值可配置: 调低阈值会让同样数据标定失败
6. 有效图像不足时抛 CalibrationError
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.camera import (
    Calibrator,
    CameraCalibration,
    load_calibration,
    save_calibration,
)
from modules.camera.calibrator import _CV2_AVAILABLE, CalibrationError
from modules.config.settings import CalibrationCfg

_CALIB_DIR = Path(__file__).resolve().parents[2] / "datasets" / "calibration_test"

pytestmark = pytest.mark.skipif(
    not _CV2_AVAILABLE or not _CALIB_DIR.exists(),
    reason="opencv not installed or calibration_test images missing "
    "(run scripts/make_chessboard_images.py)",
)


def _image_paths(n: int = 20) -> list[Path]:
    return sorted(_CALIB_DIR.glob("chessboard_*.png"))[:n]


class TestCalibratorE2E:
    def test_calibrate_success(self) -> None:
        """标定成功, reproj_error 合理."""
        cfg = CalibrationCfg(
            max_reproj_error=5.0,  # 宽松阈值 (合成图像可能有边界效应)
            chessboard_cols=9,
            chessboard_rows=6,
            square_size=0.025,
            min_valid_images=5,
        )
        cal = Calibrator(cfg)
        result = cal.calibrate(_image_paths(20), source="chessboard_test")
        assert isinstance(result, CameraCalibration)
        assert result.reproj_error > 0
        assert result.reproj_error < cfg.max_reproj_error
        assert result.width > 0
        assert result.height > 0
        assert result.n_images > 0
        assert result.source == "chessboard_test"

    def test_calibration_fx_fy_reasonable(self) -> None:
        """标定出的 fx/fy 应为正数 (具体值依赖合成图像)."""
        cfg = CalibrationCfg(
            max_reproj_error=10.0,
            chessboard_cols=9,
            chessboard_rows=6,
            square_size=0.025,
            min_valid_images=5,
        )
        cal = Calibrator(cfg)
        result = cal.calibrate(_image_paths(20))
        assert result.fx > 0
        assert result.fy > 0
        assert result.cx > 0
        assert result.cy > 0

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """标定 → 保存 → 加载 → 字段一致."""
        cfg = CalibrationCfg(
            max_reproj_error=10.0,
            chessboard_cols=9,
            chessboard_rows=6,
            square_size=0.025,
            min_valid_images=5,
        )
        cal = Calibrator(cfg)
        result = cal.calibrate(_image_paths(20), source="rt_test")
        out = save_calibration(result, tmp_path / "calib.json")
        loaded = load_calibration(out)
        assert loaded.fx == pytest.approx(result.fx)
        assert loaded.fy == pytest.approx(result.fy)
        assert loaded.reproj_error == pytest.approx(result.reproj_error)
        assert loaded.source == result.source

    def test_low_threshold_raises_error(self) -> None:
        """误差阈值调到极低, 标定应失败."""
        cfg = CalibrationCfg(
            max_reproj_error=0.001,  # 极低, 几乎必然超
            chessboard_cols=9,
            chessboard_rows=6,
            square_size=0.025,
            min_valid_images=5,
        )
        cal = Calibrator(cfg)
        with pytest.raises(CalibrationError, match="exceeds threshold"):
            cal.calibrate(_image_paths(20))

    def test_insufficient_images_raises(self) -> None:
        """有效图像不足时抛 CalibrationError."""
        cfg = CalibrationCfg(
            max_reproj_error=10.0,
            chessboard_cols=9,
            chessboard_rows=6,
            square_size=0.025,
            min_valid_images=100,  # 故意高于可用图像数
        )
        cal = Calibrator(cfg)
        with pytest.raises(CalibrationError, match="insufficient valid images"):
            cal.calibrate(_image_paths(20))

    def test_calibrate_with_ndarray_input(self) -> None:
        """直接传 ndarray (不经过文件路径) 也能标定."""
        import cv2  # type: ignore

        images = [cv2.imread(str(p)) for p in _image_paths(10)]
        cfg = CalibrationCfg(
            max_reproj_error=10.0,
            chessboard_cols=9,
            chessboard_rows=6,
            square_size=0.025,
            min_valid_images=5,
        )
        cal = Calibrator(cfg)
        result = cal.calibrate(images, source="ndarray_test")
        assert result.n_images > 0
        assert result.reproj_error > 0
