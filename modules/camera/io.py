"""M006 标定参数运行时加载/保存.

AC: "标定参数可被运行时加载".

提供:
- save_calibration(cal, path) → 写 JSON
- load_calibration(path) → 读 JSON → CameraCalibration

JSON 格式与 M005 intrinsics.json 兼容, 可直接被 Frame.intrinsics 消费.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.camera.calibration import CameraCalibration
from modules.logging import get_logger

_log = get_logger("modules.camera.io")


def save_calibration(cal: CameraCalibration, path: str | Path) -> Path:
    """保存标定结果到 JSON 文件.

    Args:
        cal: 标定结果.
        path: 输出文件路径.

    Returns:
        实际写入的路径.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(cal.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log.info(
        "calibration saved",
        path=str(out),
        reproj_error=cal.reproj_error,
        fx=cal.fx,
        fy=cal.fy,
    )
    return out


def load_calibration(path: str | Path) -> CameraCalibration:
    """从 JSON 文件加载标定参数.

    Args:
        path: JSON 文件路径.

    Returns:
        CameraCalibration 对象.

    Raises:
        FileNotFoundError: 文件不存在.
        ValueError: JSON 格式错误.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"calibration file not found: {p}")
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in {p}: {e}") from e

    cal = CameraCalibration.from_dict(data)
    _log.info(
        "calibration loaded",
        path=str(p),
        reproj_error=cal.reproj_error,
        fx=cal.fx,
        source=cal.source,
    )
    return cal


def load_calibration_to_frame_intrinsics(path: str | Path) -> dict[str, Any]:
    """加载标定并直接转成 Frame.intrinsics 格式.

    便捷方法: 一行完成 "运行时加载 → 注入 Frame".
    """
    return load_calibration(path).to_frame_intrinsics()
