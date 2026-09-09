"""M007 单元测试: equirect pixel ↔ spherical ↔ 3D ray.

覆盖:
- 已知像素点 → 已知 (yaw, pitch) (AC: 已知测试点)
- (yaw, pitch) ↔ 3D ray 双向往返
- 像素 ↔ 射线 直连映射
- 像素 → spherical → 像素 往返一致 (含边界 wrap)
- 单位长度不变
- 极点 (上下边) 处理
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
    spherical_to_pixel,
    spherical_to_ray,
)

# 典型 equirect 尺寸 (用于全部测试).
W, H = 2048, 1024

# AC: 已知测试点角度误差阈值 (deg). 严格浮点精度, 1e-6 deg.
_AC_ANGULAR_TOL_DEG = 1e-6


class TestPixelToSphericalKnownPoints:
    """AC: 已知像素点 → 已知 (yaw, pitch), 角度误差 < 阈值."""

    @pytest.mark.parametrize(
        ("u", "v", "exp_yaw_deg", "exp_pitch_deg"),
        [
            (W / 2, H / 2, 0.0, 0.0),  # 中心: 前
            (W / 4, H / 2, 90.0, 0.0),  # 左四分之一: yaw=+π/2 (左)
            (3 * W / 4, H / 2, -90.0, 0.0),  # 右四分之一: yaw=-π/2 (右)
            (W / 2, H / 4, 0.0, 45.0),  # 上四分之一: pitch=+π/4
            (W / 2, 3 * H / 4, 0.0, -45.0),  # 下四分之一: pitch=-π/4
            (0, H / 2, 180.0, 0.0),  # 左边缘: yaw=+π (后)
            (W, H / 2, -180.0, 0.0),  # 右边界: yaw=-π (与左边缘同方向, 后)
            (W / 2, 0, 0.0, 90.0),  # 顶行: pitch=+π/2 (上)
            (W / 2, H, 0.0, -90.0),  # 底行: pitch=-π/2 (下)
        ],
    )
    def test_known_points_angular_error(
        self, u: float, v: float, exp_yaw_deg: float, exp_pitch_deg: float
    ) -> None:
        yaw, pitch = pixel_to_spherical(u, v, W, H)
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        # 规范化 yaw 到 [-180, 180] 便于比较 (180 与 -180 等价).
        yaw_deg_n = ((yaw_deg + 180.0) % 360.0) - 180.0
        exp_yaw_n = ((exp_yaw_deg + 180.0) % 360.0) - 180.0
        assert abs(yaw_deg_n - exp_yaw_n) < _AC_ANGULAR_TOL_DEG, (
            f"u={u},v={v}: yaw={yaw_deg_n}°, expected={exp_yaw_n}°"
        )
        assert abs(pitch_deg - exp_pitch_deg) < _AC_ANGULAR_TOL_DEG, (
            f"u={u},v={v}: pitch={pitch_deg}°, expected={exp_pitch_deg}°"
        )


class TestSphericalToPixelRoundTrip:
    """pixel → spherical → pixel 往返一致 (含边界)."""

    @pytest.mark.parametrize(
        ("u", "v"),
        [
            (W / 2, H / 2),
            (0, H / 2),  # 左边缘 (yaw=π, wrap 到 u=0)
            (W - 1, H / 2),  # 右边缘 (yaw≈-π)
            (W / 2, 0),  # 顶
            (W / 2, H - 1),  # 底
            (W / 4, H / 4),
            (3 * W / 4, 3 * H / 4),
            (123, 456),
            (W - 2, 1),
        ],
    )
    def test_roundtrip(self, u: float, v: float) -> None:
        yaw, pitch = pixel_to_spherical(u, v, W, H)
        u2, v2 = spherical_to_pixel(yaw, pitch, W, H)
        # yaw=π 与 yaw=-π 都映射到 u=0 (后向, 像素 wrap).
        diff_u = min(abs(u2 - u), W - abs(u2 - u))
        assert diff_u < 1e-3, f"u roundtrip failed: {u} → {u2}"
        assert abs(v2 - v) < 1e-3, f"v roundtrip failed: {v} → {v2}"


class TestSphericalToRayKnown:
    """(yaw, pitch) → 3D ray 已知方向."""

    def test_forward(self) -> None:
        r = spherical_to_ray(0.0, 0.0)
        assert np.allclose(r, [1, 0, 0])

    def test_left(self) -> None:
        # yaw=+π/2 → -z (左)
        r = spherical_to_ray(math.pi / 2, 0.0)
        assert np.allclose(r, [0, 0, -1])

    def test_right(self) -> None:
        # yaw=-π/2 → +z (右)
        r = spherical_to_ray(-math.pi / 2, 0.0)
        assert np.allclose(r, [0, 0, 1])

    def test_up(self) -> None:
        r = spherical_to_ray(0.0, math.pi / 2)
        assert np.allclose(r, [0, 1, 0])

    def test_down(self) -> None:
        r = spherical_to_ray(0.0, -math.pi / 2)
        assert np.allclose(r, [0, -1, 0])

    def test_back(self) -> None:
        # yaw=±π → -x (后)
        r1 = spherical_to_ray(math.pi, 0.0)
        r2 = spherical_to_ray(-math.pi, 0.0)
        assert np.allclose(r1, [-1, 0, 0])
        assert np.allclose(r2, [-1, 0, 0])


class TestRayToSpherical:
    """3D ray → (yaw, pitch)."""

    def test_forward(self) -> None:
        yaw, pitch = ray_to_spherical([1, 0, 0])
        assert yaw == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_left(self) -> None:
        yaw, pitch = ray_to_spherical([0, 0, -1])
        assert yaw == pytest.approx(math.pi / 2, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_up(self) -> None:
        yaw, pitch = ray_to_spherical([0, 1, 0])
        assert yaw == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-9)

    def test_zero_vector_returns_zero(self) -> None:
        yaw, pitch = ray_to_spherical([0, 0, 0])
        assert yaw == 0.0
        assert pitch == 0.0

    def test_unnormalized_input(self) -> None:
        # 输入未归一化也应得到正确 (yaw, pitch).
        yaw, pitch = ray_to_spherical([5, 0, 0])
        assert yaw == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)


class TestSphericalRayRoundTrip:
    """(yaw, pitch) → ray → (yaw, pitch) 往返一致."""

    def test_roundtrip_random(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            yaw = rng.uniform(-math.pi + 0.01, math.pi - 0.01)
            pitch = rng.uniform(-math.pi / 2 + 0.01, math.pi / 2 - 0.01)
            r = spherical_to_ray(yaw, pitch)
            yaw2, pitch2 = ray_to_spherical(r)
            # yaw 可差 2π, 规范化比较.
            yaw_diff = (yaw2 - yaw + math.pi) % (2 * math.pi) - math.pi
            assert abs(yaw_diff) < 1e-9, f"yaw roundtrip: {yaw} → {yaw2}"
            assert abs(pitch2 - pitch) < 1e-9, f"pitch roundtrip: {pitch} → {pitch2}"

    def test_unit_length(self) -> None:
        rng = random.Random(7)
        for _ in range(30):
            yaw = rng.uniform(-math.pi, math.pi)
            pitch = rng.uniform(-math.pi / 2, math.pi / 2)
            r = spherical_to_ray(yaw, pitch)
            assert np.linalg.norm(r) == pytest.approx(1.0, abs=1e-9)


class TestPixelRayDirect:
    """pixel → ray 直连已知点."""

    def test_center_pixel_is_forward(self) -> None:
        r = pixel_to_ray(W / 2, H / 2, W, H)
        assert np.allclose(r, [1, 0, 0])

    def test_top_center_is_up(self) -> None:
        r = pixel_to_ray(W / 2, 0, W, H)
        assert np.allclose(r, [0, 1, 0])

    def test_bottom_center_is_down(self) -> None:
        r = pixel_to_ray(W / 2, H, W, H)
        assert np.allclose(r, [0, -1, 0])

    def test_left_edge_is_backward(self) -> None:
        r = pixel_to_ray(0, H / 2, W, H)
        assert np.allclose(r, [-1, 0, 0], atol=1e-9)

    def test_right_edge_is_backward(self) -> None:
        r = pixel_to_ray(W - 1, H / 2, W, H)
        # u=W-1 → yaw≈-π → ray≈(-1, 0, 0)
        assert np.allclose(r, [-1, 0, 0], atol=0.01)

    def test_left_quarter_is_left(self) -> None:
        r = pixel_to_ray(W / 4, H / 2, W, H)
        assert np.allclose(r, [0, 0, -1])

    def test_right_quarter_is_right(self) -> None:
        r = pixel_to_ray(3 * W / 4, H / 2, W, H)
        assert np.allclose(r, [0, 0, 1])


class TestRayToPixel:
    """3D ray → pixel 已知映射."""

    def test_forward_to_center(self) -> None:
        u, v = ray_to_pixel(np.array([1, 0, 0]), W, H)
        assert u == pytest.approx(W / 2, abs=1e-3)
        assert v == pytest.approx(H / 2, abs=1e-3)

    def test_up_to_top(self) -> None:
        u, v = ray_to_pixel(np.array([0, 1, 0]), W, H)
        assert u == pytest.approx(W / 2, abs=1e-3)
        assert v == pytest.approx(0.0, abs=1e-3)

    def test_down_to_bottom(self) -> None:
        u, v = ray_to_pixel(np.array([0, -1, 0]), W, H)
        assert u == pytest.approx(W / 2, abs=1e-3)
        # v 截断到 [0, H), 应接近 H
        assert v > H - 1e-3

    def test_left_to_quarter(self) -> None:
        u, v = ray_to_pixel(np.array([0, 0, -1]), W, H)
        assert u == pytest.approx(W / 4, abs=1e-3)
        assert v == pytest.approx(H / 2, abs=1e-3)

    def test_backward_wraps(self) -> None:
        # 后向 (-1, 0, 0) 应映射到 u=0 或 u=W (都正确, wrap).
        u, v = ray_to_pixel(np.array([-1, 0, 0]), W, H)
        assert u < 1e-3 or u > W - 1e-3
        assert v == pytest.approx(H / 2, abs=1e-3)
