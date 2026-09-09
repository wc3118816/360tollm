"""M007 集成测试: 全景投影模型端到端.

验证组合流程:
1. 像素 → 射线 → 旋转 → 像素 (已知映射)
2. R_ypr @ (1,0,0) 与 spherical_to_ray 一致 (跨模块)
3. 多个已知点 + 旋转 + 反投影的链路
4. AC 角度误差通过阈值 (随机点)
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from modules.projection import (
    pixel_to_ray,
    pixel_to_spherical,
    ray_to_pixel,
    ray_to_spherical,
    rotation_pitch,
    rotation_yaw,
    rotation_ypr,
    spherical_to_ray,
)

W, H = 2048, 1024


class TestRotatePixelToPixel:
    """像素 → 射线 → 旋转 → 射线 → 像素, 验证已知映射."""

    def test_center_rotated_left_lands_at_left_quarter(self) -> None:
        """中心 (前) → yaw 旋转 +π/2 (左) → 左四分之一像素."""
        ray = pixel_to_ray(W / 2, H / 2, W, H)
        assert np.allclose(ray, [1, 0, 0])
        rotated = rotation_yaw(math.pi / 2) @ ray
        assert np.allclose(rotated, [0, 0, -1])
        u, v = ray_to_pixel(rotated, W, H)
        assert u == pytest.approx(W / 4, abs=1e-3)
        assert v == pytest.approx(H / 2, abs=1e-3)

    def test_center_rotated_right_lands_at_right_quarter(self) -> None:
        ray = pixel_to_ray(W / 2, H / 2, W, H)
        rotated = rotation_yaw(-math.pi / 2) @ ray
        assert np.allclose(rotated, [0, 0, 1])
        u, v = ray_to_pixel(rotated, W, H)
        assert u == pytest.approx(3 * W / 4, abs=1e-3)
        assert v == pytest.approx(H / 2, abs=1e-3)

    def test_center_rotated_up_lands_at_top(self) -> None:
        ray = pixel_to_ray(W / 2, H / 2, W, H)
        rotated = rotation_pitch(math.pi / 2) @ ray
        assert np.allclose(rotated, [0, 1, 0])
        u, v = ray_to_pixel(rotated, W, H)
        assert u == pytest.approx(W / 2, abs=1e-3)
        assert v == pytest.approx(0.0, abs=1e-3)

    def test_left_quarter_rotated_back_to_center(self) -> None:
        """左四分之一 → yaw 旋转 -π/2 → 中心 (前)."""
        ray = pixel_to_ray(W / 4, H / 2, W, H)
        assert np.allclose(ray, [0, 0, -1])
        rotated = rotation_yaw(-math.pi / 2) @ ray
        assert np.allclose(rotated, [1, 0, 0])
        u, v = ray_to_pixel(rotated, W, H)
        assert u == pytest.approx(W / 2, abs=1e-3)
        assert v == pytest.approx(H / 2, abs=1e-3)


class TestYPRConsistency:
    """rotation_ypr 与 spherical_to_ray 跨模块一致."""

    @pytest.mark.parametrize(
        ("yaw", "pitch", "roll"),
        [
            (0.0, 0.0, 0.0),
            (0.5, 0.3, 0.0),
            (-1.0, 0.5, 0.0),
            (math.pi / 3, -math.pi / 4, 0.0),
        ],
    )
    def test_ypr_forward_equals_spherical(self, yaw: float, pitch: float, roll: float) -> None:
        R = rotation_ypr(yaw, pitch, roll)
        forward = R @ np.array([1, 0, 0])
        expected = spherical_to_ray(yaw, pitch)
        assert np.allclose(forward, expected, atol=1e-10)

    def test_roll_does_not_change_forward(self) -> None:
        """roll 绕前向轴旋转, 不改变前向方向."""
        rng = random.Random(7)
        for _ in range(10):
            yaw = rng.uniform(-math.pi, math.pi)
            pitch = rng.uniform(-math.pi / 2 + 0.01, math.pi / 2 - 0.01)
            roll = rng.uniform(-math.pi, math.pi)
            R = rotation_ypr(yaw, pitch, roll)
            forward = R @ np.array([1, 0, 0])
            expected = spherical_to_ray(yaw, pitch)
            assert np.allclose(forward, expected, atol=1e-10)


class TestAngularErrorThreshold:
    """AC: 已知测试点的角度误差通过阈值.

    对随机 (yaw, pitch) → ray → (yaw2, pitch2) 往返, 角度误差 < 1e-6 deg.
    """

    ANGULAR_TOL_DEG = 1e-6

    def test_random_roundtrip_angular_error(self) -> None:
        rng = random.Random(2024)
        max_err = 0.0
        for _ in range(100):
            yaw = rng.uniform(-math.pi + 0.01, math.pi - 0.01)
            pitch = rng.uniform(-math.pi / 2 + 0.01, math.pi / 2 - 0.01)
            r = spherical_to_ray(yaw, pitch)
            yaw2, pitch2 = ray_to_spherical(r)
            yaw_err = math.degrees(abs((yaw2 - yaw + math.pi) % (2 * math.pi) - math.pi))
            pitch_err = math.degrees(abs(pitch2 - pitch))
            max_err = max(max_err, yaw_err, pitch_err)
        assert max_err < self.ANGULAR_TOL_DEG, (
            f"max angular error {max_err}° > {self.ANGULAR_TOL_DEG}°"
        )


class TestFullPipelineRoundTrip:
    """完整链路: pixel → spherical → ray → spherical → pixel."""

    @pytest.mark.parametrize(
        ("u", "v"),
        [
            (W / 2, H / 2),
            (W / 4, H / 4),
            (3 * W / 4, 3 * H / 4),
            (100, 200),
            (W - 100, H - 200),
        ],
    )
    def test_roundtrip(self, u: float, v: float) -> None:
        yaw, pitch = pixel_to_spherical(u, v, W, H)
        r = spherical_to_ray(yaw, pitch)
        yaw2, pitch2 = ray_to_spherical(r)
        u2, v2 = ray_to_pixel(r, W, H)
        # 像素往返
        diff_u = min(abs(u2 - u), W - abs(u2 - u))
        assert diff_u < 1e-3, f"u: {u} → {u2}"
        assert abs(v2 - v) < 1e-3, f"v: {v} → {v2}"
        # 球面坐标往返
        yaw_diff = abs((yaw2 - yaw + math.pi) % (2 * math.pi) - math.pi)
        assert yaw_diff < 1e-9
        assert abs(pitch2 - pitch) < 1e-9
