"""M007 yaw/pitch/roll 旋转矩阵.

约定 (右手坐标系, 与 spherical.py 一致):
    +x = forward, +y = up, +z = right

基本旋转 (绕世界轴):
    R_yaw(yaw)   = R_y(yaw)   绕 +y 轴, 把 +x 转向 -z (即 yaw=+π/2 = 左)
    R_pitch(pitch) = R_z(pitch) 绕 +z 轴, 把 +x 转向 +y (即 pitch=+π/2 = 上)
    R_roll(roll)  = R_x(roll)  绕 +x 轴, 把 +y 转向 +z (即 roll=+π/2 = 右翻)

组合 R_ypr(yaw, pitch, roll) = R_yaw(yaw) @ R_pitch(pitch) @ R_roll(roll),
对应 intrinsic Y, P, R (先 yaw 绕 body-y, 再 pitch 绕 body-z, 最后 roll 绕 body-x).
当 roll=0 时:
    R_ypr(yaw, pitch, 0) @ (1, 0, 0) == spherical_to_ray(yaw, pitch)
即 "look at (yaw, pitch)" 的前向方向.
"""

from __future__ import annotations

import math

import numpy as np


def rotation_yaw(yaw: float) -> np.ndarray:
    """绕 +y 轴的旋转矩阵 (3x3, float64)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_pitch(pitch: float) -> np.ndarray:
    """绕 +z 轴的旋转矩阵 (3x3, float64)."""
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_roll(roll: float) -> np.ndarray:
    """绕 +x 轴的旋转矩阵 (3x3, float64)."""
    c, s = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_ypr(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """组合旋转: R = R_yaw(yaw) @ R_pitch(pitch) @ R_roll(roll).

    对应 intrinsic Y, P, R 顺序 (先绕 body-y, 再绕 body-z, 再绕 body-x).
    矩阵作用于向量 v 时: v' = R @ v, 即 R_roll 先应用到 v.

    当 roll=0:
        R @ (1, 0, 0) = (cos pitch * cos yaw, sin pitch, -cos pitch * sin yaw)
        与 spherical_to_ray(yaw, pitch) 一致.

    Args:
        yaw: 方位角 (rad), 绕 +y.
        pitch: 俯仰角 (rad), 绕 +z.
        roll: 翻滚角 (rad), 绕 +x. 默认 0.

    Returns:
        3x3 旋转矩阵 (float64).
    """
    return rotation_yaw(yaw) @ rotation_pitch(pitch) @ rotation_roll(roll)
