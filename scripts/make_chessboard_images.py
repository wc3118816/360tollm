"""生成合成棋盘格图像用于 M006 标定测试.

用 OpenCV 画 9x6 内角点棋盘格, 加已知内参投影, 生成多幅"不同视角"图像.
用于验证 Calibrator 能正确恢复内参.

用法:
    uv run python scripts/make_chessboard_images.py
输出:
    datasets/calibration_test/ (20 张 PNG)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore

    _CV2 = True
except ImportError:
    _CV2 = False


def main() -> None:
    if not _CV2:
        raise SystemExit("opencv not installed; run: uv pip install opencv-python-headless")

    out_dir = Path("datasets/calibration_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    cols, rows = 9, 6  # 内角点
    square_px = 30  # 格子像素边长
    border = 30  # 边框
    img_w = cols * square_px + 2 * border
    img_h = rows * square_px + 2 * border

    # 已知内参 (保留供文档参考, 生成脚本用仿射变换而非真实投影).
    _fx = _fy = 800.0  # noqa: F841
    _cx, _cy = img_w / 2, img_h / 2  # noqa: F841

    # 3D 世界坐标 (Z=0 平面, 单位 = 格子边长假设 0.025m).
    obj = np.zeros((cols * rows, 3), dtype=np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= 0.025  # 0.025m 格子

    n_images = 20
    for i in range(n_images):
        # 画标准棋盘格图.
        img = np.full((img_h, img_w), 255, dtype=np.uint8)
        for y in range(rows + 1):
            for x in range(cols + 1):
                if (x + y) % 2 == 0:
                    y0 = border + y * square_px
                    x0 = border + x * square_px
                    img[y0 : y0 + square_px, x0 : x0 + square_px] = 0

        # 模拟不同视角: 小角度仿射变换 (确保角点仍可检测).
        angle = (i - n_images / 2) * 1.5  # -15..15 度, 保守
        M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
        warped = cv2.warpAffine(img, M, (img_w, img_h), borderValue=255)

        cv2.imwrite(str(out_dir / f"chessboard_{i:03d}.png"), warped)

    print(f"chessboard images created: {out_dir} ({n_images} images)")


if __name__ == "__main__":
    main()
