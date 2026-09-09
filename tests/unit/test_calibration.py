"""modules.camera.calibration 单测 (M006).

覆盖:
- CameraCalibration 构造 + 属性访问 (fx/fy/cx/cy)
- camera_matrix 形状校验
- reproj_error 负值拒绝
- dist_coeffs 不足 5 个时补零
- to_dict / from_dict 往返一致
- to_frame_intrinsics 输出格式正确
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.camera.calibration import CameraCalibration


def _make_calibration(
    fx: float = 800.0,
    fy: float = 800.0,
    cx: float = 320.0,
    cy: float = 240.0,
    reproj: float = 0.3,
) -> CameraCalibration:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([0.1, -0.05, 0.001, 0.002, 0.0], dtype=np.float64)
    return CameraCalibration(
        camera_matrix=K,
        dist_coeffs=dist,
        reproj_error=reproj,
        width=640,
        height=480,
        n_images=10,
        n_corners=(9, 6),
        square_size=0.025,
        source="test",
    )


def test_construction_and_properties() -> None:
    cal = _make_calibration()
    assert cal.fx == 800.0
    assert cal.fy == 800.0
    assert cal.cx == 320.0
    assert cal.cy == 240.0
    assert cal.width == 640
    assert cal.height == 480
    assert cal.reproj_error == 0.3
    assert cal.source == "test"


def test_camera_matrix_shape_validated() -> None:
    """非 (3,3) 矩阵抛 ValueError."""
    bad_K = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="camera_matrix must be"):
        CameraCalibration(
            camera_matrix=bad_K,
            dist_coeffs=np.zeros(5),
            reproj_error=0.0,
            width=640,
            height=480,
        )


def test_reproj_error_negative_rejected() -> None:
    """负的重投影误差抛 ValueError."""
    K = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match="reproj_error must be >= 0"):
        CameraCalibration(
            camera_matrix=K,
            dist_coeffs=np.zeros(5),
            reproj_error=-0.1,
            width=640,
            height=480,
        )


def test_dist_coeffs_padded_to_5() -> None:
    """dist_coeffs 不足 5 个时补零."""
    K = np.eye(3, dtype=np.float64)
    cal = CameraCalibration(
        camera_matrix=K,
        dist_coeffs=np.array([0.1, 0.2]),  # 只有 2 个
        reproj_error=0.0,
        width=640,
        height=480,
    )
    assert cal.dist_coeffs.size == 5
    assert cal.dist_coeffs[0] == 0.1
    assert cal.dist_coeffs[1] == 0.2
    assert cal.dist_coeffs[2] == 0.0


def test_to_dict_and_from_dict_roundtrip() -> None:
    """to_dict + from_dict 往返一致."""
    cal = _make_calibration(reproj=0.42)
    d = cal.to_dict()
    cal2 = CameraCalibration.from_dict(d)
    assert cal2.fx == cal.fx
    assert cal2.fy == cal.fy
    assert cal2.cx == cal.cx
    assert cal2.cy == cal.cy
    assert cal2.reproj_error == cal.reproj_error
    assert cal2.width == cal.width
    assert cal2.height == cal.height
    assert cal2.source == cal.source
    np.testing.assert_allclose(cal2.camera_matrix, cal.camera_matrix)
    np.testing.assert_allclose(cal2.dist_coeffs, cal.dist_coeffs)


def test_to_frame_intrinsics_format() -> None:
    """to_frame_intrinsics 输出与 M005 intrinsics.json 格式一致."""
    cal = _make_calibration()
    intr = cal.to_frame_intrinsics()
    assert intr["type"] == "perspective"
    assert intr["fx"] == 800.0
    assert intr["fy"] == 800.0
    assert intr["cx"] == 320.0
    assert intr["cy"] == 240.0
    assert intr["width"] == 640
    assert intr["height"] == 480
    assert "distortion" in intr
    assert intr["reproj_error"] == 0.3
    assert len(intr["distortion"]) == 5


def test_to_dict_json_serializable() -> None:
    """to_dict 结果可直接 json.dumps (无 ndarray)."""
    import json

    cal = _make_calibration()
    d = cal.to_dict()
    s = json.dumps(d)  # 不抛异常即 OK
    assert isinstance(s, str)
