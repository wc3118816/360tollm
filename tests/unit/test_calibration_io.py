"""modules.camera.io 单测 (M006).

覆盖:
- save_calibration + load_calibration 往返一致
- load_calibration 文件不存在抛 FileNotFoundError
- load_calibration 无效 JSON 抛 ValueError
- load_calibration_to_frame_intrinsics 直接返回 Frame.intrinsics 格式
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modules.camera.calibration import CameraCalibration
from modules.camera.io import (
    load_calibration,
    load_calibration_to_frame_intrinsics,
    save_calibration,
)


def _make_cal() -> CameraCalibration:
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], dtype=np.float64)
    return CameraCalibration(
        camera_matrix=K,
        dist_coeffs=np.array([0.1, -0.05, 0.001, 0.002, 0.0], dtype=np.float64),
        reproj_error=0.35,
        width=640,
        height=480,
        n_images=15,
        source="test_io",
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """save → load 往返一致."""
    cal = _make_cal()
    out = save_calibration(cal, tmp_path / "calib.json")
    assert out.exists()

    cal2 = load_calibration(out)
    assert cal2.fx == cal.fx
    assert cal2.fy == cal.fy
    assert cal2.cx == cal.cx
    assert cal2.cy == cal.cy
    assert cal2.reproj_error == cal.reproj_error
    assert cal2.source == cal.source
    np.testing.assert_allclose(cal2.camera_matrix, cal.camera_matrix)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="calibration file not found"):
        load_calibration(tmp_path / "nonexistent.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_calibration(p)


def test_load_to_frame_intrinsics(tmp_path: Path) -> None:
    """load_calibration_to_frame_intrinsics 返回 Frame.intrinsics 格式."""
    cal = _make_cal()
    out = save_calibration(cal, tmp_path / "calib.json")
    intr = load_calibration_to_frame_intrinsics(out)
    assert intr["type"] == "perspective"
    assert intr["fx"] == 800.0
    assert intr["reproj_error"] == 0.35
    assert "distortion" in intr


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """save 自动创建父目录."""
    cal = _make_cal()
    out = save_calibration(cal, tmp_path / "sub" / "dir" / "calib.json")
    assert out.exists()
    assert out.parent.is_dir()
