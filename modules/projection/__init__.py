"""M007 全景投影模型: equirectangular ↔ spherical ↔ 3D ray.

公开 API:

球面坐标变换 (spherical.py):
    pixel_to_spherical(u, v, W, H) -> (yaw, pitch)
    spherical_to_pixel(yaw, pitch, W, H) -> (u, v)
    spherical_to_ray(yaw, pitch) -> np.ndarray (3,) unit vector
    ray_to_spherical(ray) -> (yaw, pitch)
    pixel_to_ray(u, v, W, H) -> np.ndarray (3,)
    ray_to_pixel(ray, W, H) -> (u, v)

旋转矩阵 (rotation.py):
    rotation_yaw(yaw) -> (3,3)
    rotation_pitch(pitch) -> (3,3)
    rotation_roll(roll) -> (3,3)
    rotation_ypr(yaw, pitch, roll=0) -> (3,3)

FOV (fov.py):
    perspective_hfov(fx, width) / perspective_vfov(fy, height) / perspective_fov(...)
    focal_from_hfov(hfov, width) / focal_from_vfov(vfov, height)
    equirect_full_hfov() / equirect_full_vfov()
    equirect_region_angular_extent(u_min, u_max, v_min, v_max, W, H) -> (Δyaw, Δpitch)
    equirect_region_solid_angle(...) -> steradians
"""

from __future__ import annotations

from .fov import (
    equirect_full_hfov,
    equirect_full_vfov,
    equirect_region_angular_extent,
    equirect_region_solid_angle,
    focal_from_hfov,
    focal_from_vfov,
    perspective_fov,
    perspective_hfov,
    perspective_vfov,
)
from .rotation import (
    rotation_pitch,
    rotation_roll,
    rotation_yaw,
    rotation_ypr,
)
from .spherical import (
    pixel_to_ray,
    pixel_to_spherical,
    ray_to_pixel,
    ray_to_spherical,
    spherical_to_pixel,
    spherical_to_ray,
)

__all__ = [
    # spherical
    "pixel_to_spherical",
    "spherical_to_pixel",
    "spherical_to_ray",
    "ray_to_spherical",
    "pixel_to_ray",
    "ray_to_pixel",
    # rotation
    "rotation_yaw",
    "rotation_pitch",
    "rotation_roll",
    "rotation_ypr",
    # fov
    "perspective_hfov",
    "perspective_vfov",
    "perspective_fov",
    "focal_from_hfov",
    "focal_from_vfov",
    "equirect_full_hfov",
    "equirect_full_vfov",
    "equirect_region_angular_extent",
    "equirect_region_solid_angle",
]
