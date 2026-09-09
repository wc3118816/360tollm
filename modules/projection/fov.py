"""M007 FOV (视场角) 计算.

两类:
1. 透视相机 (pinhole): 由 fx/fy + 图像尺寸算 H/V FOV.
2. Equirectangular: 整图 FOV 固定 (2π × π); 子区域可算角范围与立体角.
"""

from __future__ import annotations

import math


def perspective_hfov(fx: float, width: int) -> float:
    """水平 FOV (rad) 从焦距 + 图像宽度.

    hFOV = 2 * atan(W / (2*fx))
    """
    return 2.0 * math.atan(width / (2.0 * fx))


def perspective_vfov(fy: float, height: int) -> float:
    """垂直 FOV (rad) 从焦距 + 图像高度."""
    return 2.0 * math.atan(height / (2.0 * fy))


def perspective_fov(fx: float, fy: float, width: int, height: int) -> tuple[float, float]:
    """同时返回 (hFOV, vFOV), 单位 rad."""
    return perspective_hfov(fx, width), perspective_vfov(fy, height)


def focal_from_hfov(hfov: float, width: int) -> float:
    """hFOV (rad) → 焦距 fx. perspective_hfov 的反函数."""
    return (width / 2.0) / math.tan(hfov / 2.0)


def focal_from_vfov(vfov: float, height: int) -> float:
    """vFOV (rad) → 焦距 fy."""
    return (height / 2.0) / math.tan(vfov / 2.0)


def equirect_full_hfov() -> float:
    """完整 equirect 水平 FOV = 2π (360°)."""
    return 2.0 * math.pi


def equirect_full_vfov() -> float:
    """完整 equirect 垂直 FOV = π (180°)."""
    return math.pi


def equirect_region_angular_extent(  # noqa: PLR0917
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    W: int,
    H: int,
) -> tuple[float, float]:
    """equirect 矩形子区域的角范围 (Δyaw, Δpitch), 单位 rad.

    简单线性映射: Δyaw = (u_max - u_min) / W * 2π, Δpitch 同理.
    注意这是 equirect 像素空间的线性范围, 不是真正的 3D 角度 (在极点附近会失真).
    """
    dyaw = (u_max - u_min) / W * 2.0 * math.pi
    dpitch = (v_max - v_min) / H * math.pi
    return dyaw, dpitch


def equirect_region_solid_angle(  # noqa: PLR0917
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    W: int,
    H: int,
) -> float:
    """equirect 矩形子区域的立体角 (steradians).

    对球面积分: dΩ = cos(pitch) * dpitch * dyaw.
    矩形区域 [u_min, u_max] × [v_min, v_max] 对应:
        yaw ∈ [yaw_min, yaw_max], pitch ∈ [pitch_min, pitch_max]
        Ω = (sin(pitch_max) - sin(pitch_min)) * (yaw_max - yaw_min)

    完整 equirect (u: 0..W, v: 0..H) → Ω = (sin(π/2) - sin(-π/2)) * 2π = 4π (整球).
    """
    yaw_min = (0.5 - u_max / W) * 2.0 * math.pi
    yaw_max = (0.5 - u_min / W) * 2.0 * math.pi
    pitch_min = (0.5 - v_max / H) * math.pi
    pitch_max = (0.5 - v_min / H) * math.pi
    return (math.sin(pitch_max) - math.sin(pitch_min)) * (yaw_max - yaw_min)
