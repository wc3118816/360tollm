"""M007 全景投影模型: equirectangular ↔ spherical ↔ 3D ray.

约定 (右手坐标系):
    +x = forward (yaw=0, pitch=0 时的方向)
    +y = up
    +z = right (面向 +x 时的右手方向)

球面坐标:
    yaw   ∈ [-π, π]     方位角 (绕 +y 轴), 0 = +x (前), +π/2 = -z (左), -π/2 = +z (右)
    pitch ∈ [-π/2, π/2]  俯仰角 (绕 +z 轴), 0 = 地平, +π/2 = +y (上), -π/2 = -y (下)

Equirectangular 图像 (W x H):
    u ∈ [0, W), v ∈ [0, H)
    u = W/2 → yaw = 0       (中心列 = 前)
    u = 0   → yaw = +π      (左边缘 = 后, 与右边缘 -π 同方向)
    u = W   → yaw = -π      (右边缘 = 后)
    v = H/2 → pitch = 0     (中心行 = 地平)
    v = 0   → pitch = +π/2  (顶行 = 上)
    v = H   → pitch = -π/2  (底行 = 下)

3D 单位射线 r = (x, y, z) 与 (yaw, pitch) 的关系:
    r = R_yaw(yaw) @ R_z(pitch) @ (1, 0, 0)
      = (cos(pitch) * cos(yaw), sin(pitch), -cos(pitch) * sin(yaw))

验证:
    yaw=0, pitch=0           → (1, 0, 0)   前
    yaw=+π/2, pitch=0        → (0, 0, -1)  左
    yaw=-π/2, pitch=0       → (0, 0, 1)   右
    yaw=0, pitch=+π/2       → (0, 1, 0)   上
    yaw=0, pitch=-π/2       → (0, -1, 0)  下
    yaw=±π, pitch=0         → (-1, 0, 0)  后
"""

from __future__ import annotations

import math

import numpy as np

# 数值安全阈值: 用于射线归一化与 pitch 截断.
_EPS = 1e-12


def pixel_to_spherical(u: float, v: float, W: int, H: int) -> tuple[float, float]:
    """Equirect 像素 (u, v) → (yaw, pitch), 单位弧度.

    Args:
        u: 列坐标, [0, W). u=W/2 → yaw=0; u=0 → yaw=+π; u=W → yaw=-π.
        v: 行坐标, [0, H). v=H/2 → pitch=0; v=0 → pitch=+π/2; v=H → pitch=-π/2.
        W: 图像宽度 (像素).
        H: 图像高度 (像素).

    Returns:
        (yaw, pitch) in radians. yaw ∈ [-π, π], pitch ∈ [-π/2, π/2].
    """
    yaw = (0.5 - u / W) * 2.0 * math.pi
    pitch = (0.5 - v / H) * math.pi
    return yaw, pitch


def spherical_to_pixel(yaw: float, pitch: float, W: int, H: int) -> tuple[float, float]:
    """(yaw, pitch) → equirect 像素 (u, v). pixel_to_spherical 的逆.

    Args:
        yaw: 方位角 (rad), 任意值 (会被 wrap 到 [-π, π] 对应的像素列).
        pitch: 俯仰角 (rad), 期望 ∈ [-π/2, π/2]; 越界会被截断.
        W: 图像宽度.
        H: 图像高度.

    Returns:
        (u, v), u ∈ [0, W), v ∈ [0, H].
    """
    u = (0.5 - yaw / (2.0 * math.pi)) * W
    v = (0.5 - pitch / math.pi) * H
    # u wrap 到 [0, W)
    u = u % W
    # v 截断到 [0, H)
    v = max(0.0, min(H - _EPS, v))
    return u, v


def spherical_to_ray(yaw: float, pitch: float) -> np.ndarray:
    """(yaw, pitch) → 3D 单位射线 (x, y, z).

    见模块 docstring 的约定. 输出长度恒为 1 (除极点附近浮点误差外).
    """
    cp = math.cos(pitch)
    return np.array(
        [
            cp * math.cos(yaw),
            math.sin(pitch),
            -cp * math.sin(yaw),
        ],
        dtype=np.float64,
    )


def ray_to_spherical(ray: np.ndarray) -> tuple[float, float]:
    """3D 射线 → (yaw, pitch). 不要求输入归一化.

    Args:
        ray: 任意 3D 向量 (np.ndarray 或 list).

    Returns:
        (yaw, pitch). yaw ∈ [-π, π], pitch ∈ [-π/2, π/2].
        零向量返回 (0, 0).
    """
    r = np.asarray(ray, dtype=np.float64).ravel()
    n = float(np.linalg.norm(r))
    if n < _EPS:
        return 0.0, 0.0
    x, y, z = r / n
    # pitch = asin(y), 截断以防浮点误差让 |y| > 1.
    pitch = math.asin(max(-1.0, min(1.0, y)))
    # yaw = atan2(-z, x) (与 spherical_to_ray 的符号约定一致).
    yaw = math.atan2(-z, x)
    return yaw, pitch


def pixel_to_ray(u: float, v: float, W: int, H: int) -> np.ndarray:
    """Equirect 像素 → 3D 单位射线. 等价于 pixel_to_spherical + spherical_to_ray."""
    yaw, pitch = pixel_to_spherical(u, v, W, H)
    return spherical_to_ray(yaw, pitch)


def ray_to_pixel(ray: np.ndarray, W: int, H: int) -> tuple[float, float]:
    """3D 射线 → equirect 像素. 等价于 ray_to_spherical + spherical_to_pixel."""
    yaw, pitch = ray_to_spherical(ray)
    return spherical_to_pixel(yaw, pitch, W, H)
