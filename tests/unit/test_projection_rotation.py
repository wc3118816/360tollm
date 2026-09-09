"""M007 单元测试: yaw/pitch/roll 旋转矩阵.

覆盖:
- 单位角 → 单位矩阵
- 已知 90° 旋转的向量变换
- 正交性 (R @ R^T = I)
- 行列式 = 1
- 逆 = 转置
- rotation_ypr 组合顺序
- R_ypr(yaw, pitch, 0) @ (1,0,0) == spherical_to_ray(yaw, pitch) (一致性)
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from modules.projection import (
    rotation_pitch,
    rotation_roll,
    rotation_yaw,
    rotation_ypr,
)
from modules.projection.spherical import spherical_to_ray


class TestRotationYaw:
    def test_identity_at_zero(self) -> None:
        assert np.allclose(rotation_yaw(0.0), np.eye(3))

    def test_90_deg_forward_to_left(self) -> None:
        R = rotation_yaw(math.pi / 2)
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 0, -1])

    def test_minus_90_forward_to_right(self) -> None:
        R = rotation_yaw(-math.pi / 2)
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 0, 1])

    def test_180_forward_to_backward(self) -> None:
        R = rotation_yaw(math.pi)
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [-1, 0, 0])

    def test_y_unchanged(self) -> None:
        # 绕 y 轴旋转不应改变 y 分量.
        R = rotation_yaw(0.7)
        v = R @ np.array([0, 1, 0])
        assert np.allclose(v, [0, 1, 0])


class TestRotationPitch:
    def test_identity_at_zero(self) -> None:
        assert np.allclose(rotation_pitch(0.0), np.eye(3))

    def test_90_deg_forward_to_up(self) -> None:
        R = rotation_pitch(math.pi / 2)
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 1, 0])

    def test_minus_90_forward_to_down(self) -> None:
        R = rotation_pitch(-math.pi / 2)
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, -1, 0])

    def test_z_unchanged(self) -> None:
        # 绕 z 轴旋转不应改变 z 分量.
        R = rotation_pitch(0.5)
        v = R @ np.array([0, 0, 1])
        assert np.allclose(v, [0, 0, 1])


class TestRotationRoll:
    def test_identity_at_zero(self) -> None:
        assert np.allclose(rotation_roll(0.0), np.eye(3))

    def test_90_deg_up_to_right(self) -> None:
        # 绕 +x 旋转 π/2: +y → +z
        R = rotation_roll(math.pi / 2)
        v = R @ np.array([0, 1, 0])
        assert np.allclose(v, [0, 0, 1])

    def test_x_unchanged(self) -> None:
        R = rotation_roll(1.2)
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [1, 0, 0])


class TestOrthogonality:
    """旋转矩阵基本性质: 正交 + 行列式 = 1."""

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, 1.0, math.pi / 2, math.pi, 2.5])
    def test_orthogonal(self, angle: float) -> None:
        for R in [rotation_yaw(angle), rotation_pitch(angle), rotation_roll(angle)]:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    @pytest.mark.parametrize("angle", [0.0, 0.3, 1.0, math.pi / 2, math.pi])
    def test_determinant_one(self, angle: float) -> None:
        for R in [rotation_yaw(angle), rotation_pitch(angle), rotation_roll(angle)]:
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_inverse_is_transpose(self) -> None:
        rng = random.Random(123)
        for _ in range(10):
            angle = rng.uniform(-math.pi, math.pi)
            for R in [rotation_yaw(angle), rotation_pitch(angle), rotation_roll(angle)]:
                assert np.allclose(np.linalg.inv(R), R.T, atol=1e-10)


class TestRotationYPR:
    """rotation_ypr 组合."""

    def test_identity_when_all_zero(self) -> None:
        assert np.allclose(rotation_ypr(0, 0, 0), np.eye(3))

    def test_only_yaw(self) -> None:
        R = rotation_ypr(0.5, 0, 0)
        assert np.allclose(R, rotation_yaw(0.5))

    def test_only_pitch(self) -> None:
        R = rotation_ypr(0, 0.5, 0)
        assert np.allclose(R, rotation_pitch(0.5))

    def test_only_roll(self) -> None:
        R = rotation_ypr(0, 0, 0.5)
        assert np.allclose(R, rotation_roll(0.5))

    def test_matrix_product_order(self) -> None:
        """R_ypr = R_yaw @ R_pitch @ R_roll (矩阵乘法顺序)."""
        yaw, pitch, roll = 0.3, 0.4, 0.5
        R_combined = rotation_ypr(yaw, pitch, roll)
        R_manual = rotation_yaw(yaw) @ rotation_pitch(pitch) @ rotation_roll(roll)
        assert np.allclose(R_combined, R_manual)

    def test_orthogonal_combined(self) -> None:
        R = rotation_ypr(0.5, 0.3, 0.2)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


class TestConsistencyWithSpherical:
    """AC 关键: rotation_ypr 与 spherical_to_ray 一致.

    R_ypr(yaw, pitch, 0) @ (1, 0, 0) 必须等于 spherical_to_ray(yaw, pitch).
    保证 "旋转得到的方向" 与 "球面坐标得到的射线" 是同一个.
    """

    @pytest.mark.parametrize(
        ("yaw", "pitch"),
        [
            (0.0, 0.0),
            (math.pi / 2, 0.0),
            (-math.pi / 2, 0.0),
            (0.0, math.pi / 2),
            (0.0, -math.pi / 2),
            (math.pi, 0.0),
            (0.3, 0.4),
            (-1.2, 0.5),
            (2.0, -0.7),
            (-0.5, 1.0),
        ],
    )
    def test_forward_matches_spherical(self, yaw: float, pitch: float) -> None:
        R = rotation_ypr(yaw, pitch, 0.0)
        forward = R @ np.array([1, 0, 0])
        expected = spherical_to_ray(yaw, pitch)
        assert np.allclose(forward, expected), (
            f"yaw={yaw}, pitch={pitch}: forward={forward}, expected={expected}"
        )

    def test_random_consistency(self) -> None:
        rng = random.Random(99)
        for _ in range(50):
            yaw = rng.uniform(-math.pi, math.pi)
            pitch = rng.uniform(-math.pi / 2 + 0.01, math.pi / 2 - 0.01)
            R = rotation_ypr(yaw, pitch, 0.0)
            forward = R @ np.array([1, 0, 0])
            expected = spherical_to_ray(yaw, pitch)
            assert np.allclose(forward, expected, atol=1e-10)
