"""M007 单元测试: FOV 计算.

覆盖:
- 透视相机 hfov/vfov (90° 已知点)
- focal_from_hfov 反函数
- equirect 整图 FOV (2π × π)
- equirect 子区域角范围 (线性)
- equirect 子区域立体角 (整图 = 4π, 赤道带 cos(0)=1)
"""

from __future__ import annotations

import math

import pytest

from modules.projection import (
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


class TestPerspectiveFOV:
    def test_hfov_90_degrees(self) -> None:
        # fx=320, W=640 → hFOV = 2*atan(640/(2*320)) = 2*atan(1) = π/2 = 90°
        fov = perspective_hfov(320.0, 640)
        assert math.degrees(fov) == pytest.approx(90.0, abs=1e-6)

    def test_vfov_90_degrees(self) -> None:
        fov = perspective_vfov(240.0, 480)
        assert math.degrees(fov) == pytest.approx(90.0, abs=1e-6)

    def test_combined(self) -> None:
        h, v = perspective_fov(320.0, 240.0, 640, 480)
        assert math.degrees(h) == pytest.approx(90.0, abs=1e-6)
        assert math.degrees(v) == pytest.approx(90.0, abs=1e-6)

    def test_narrow_fov_with_large_fx(self) -> None:
        # fx → ∞ 时 FOV → 0
        fov = perspective_hfov(10000.0, 640)
        assert fov < math.radians(5.0)

    def test_inverse_hfov(self) -> None:
        fx = focal_from_hfov(math.radians(90.0), 640)
        assert fx == pytest.approx(320.0, abs=1e-6)

    def test_inverse_vfov(self) -> None:
        fy = focal_from_vfov(math.radians(60.0), 480)
        # 反算回去
        assert math.degrees(perspective_vfov(fy, 480)) == pytest.approx(60.0, abs=1e-6)


class TestEquirectFullFOV:
    def test_full_hfov_is_2pi(self) -> None:
        assert equirect_full_hfov() == pytest.approx(2 * math.pi)

    def test_full_vfov_is_pi(self) -> None:
        assert equirect_full_vfov() == pytest.approx(math.pi)

    def test_full_degrees(self) -> None:
        assert math.degrees(equirect_full_hfov()) == pytest.approx(360.0)
        assert math.degrees(equirect_full_vfov()) == pytest.approx(180.0)


class TestEquirectRegionExtent:
    W, H = 2048, 1024

    def test_half_image_horizontal(self) -> None:
        # 左半图 u:[0, W/2], v:[0, H]: Δyaw=π, Δpitch=π
        dyaw, dpitch = equirect_region_angular_extent(0, self.W / 2, 0, self.H, self.W, self.H)
        assert dyaw == pytest.approx(math.pi, abs=1e-9)
        assert dpitch == pytest.approx(math.pi, abs=1e-9)

    def test_quarter_image(self) -> None:
        # u:[0, W/4], v:[0, H/2]: Δyaw=π/2, Δpitch=π/2
        dyaw, dpitch = equirect_region_angular_extent(0, self.W / 4, 0, self.H / 2, self.W, self.H)
        assert dyaw == pytest.approx(math.pi / 2, abs=1e-9)
        assert dpitch == pytest.approx(math.pi / 2, abs=1e-9)

    def test_full_image(self) -> None:
        dyaw, dpitch = equirect_region_angular_extent(0, self.W, 0, self.H, self.W, self.H)
        assert dyaw == pytest.approx(2 * math.pi, abs=1e-9)
        assert dpitch == pytest.approx(math.pi, abs=1e-9)


class TestEquirectSolidAngle:
    W, H = 2048, 1024

    def test_full_image_is_4pi(self) -> None:
        # 整个 equirect 覆盖整球, 立体角 = 4π
        omega = equirect_region_solid_angle(0, self.W, 0, self.H, self.W, self.H)
        assert omega == pytest.approx(4 * math.pi, abs=1e-6)

    def test_equator_strip(self) -> None:
        # 赤道带: v ∈ [H/2 - 1, H/2 + 1] (1 像素上下), u ∈ [0, W]
        # Δpitch = 2/H * π, cos(pitch)≈1 (赤道附近)
        # Ω ≈ 2π * (2π/H)
        omega = equirect_region_solid_angle(
            0, self.W, self.H / 2 - 1, self.H / 2 + 1, self.W, self.H
        )
        expected = 2 * math.pi * (2 * math.pi / self.H)
        assert omega == pytest.approx(expected, abs=1e-3)

    def test_pole_region_smaller(self) -> None:
        # 同样的像素面积, 靠近极点的立体角应小于赤道附近 (cos(pitch) 效应).
        equator = equirect_region_solid_angle(
            0, self.W / 4, self.H / 2 - 10, self.H / 2 + 10, self.W, self.H
        )
        pole = equirect_region_solid_angle(
            0,
            self.W / 4,
            10,
            30,
            self.W,
            self.H,  # 靠近顶 (pitch 接近 +π/2)
        )
        assert pole < equator

    def test_zero_height_region(self) -> None:
        # v_min == v_max → 立体角 = 0
        omega = equirect_region_solid_angle(0, self.W, self.H / 2, self.H / 2, self.W, self.H)
        assert omega == pytest.approx(0.0, abs=1e-12)
