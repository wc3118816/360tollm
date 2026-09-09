"""M029 单元测试: WorldModel API.

覆盖:
- world_to_camera / camera_to_world 双向变换 (AC)
- 坐标变换 roundtrip 可逆
- object_pose / object_in_world / object_in_camera
- object_distance / object_direction
- distance_to_camera / direction_from_camera
- find_nearest
- SI 单位验证 (AC)
- 空模型
- 无效输入
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.objects import Object3D, Object3DList
from modules.vo.types import Pose
from modules.world_model import WorldModel


def _make_pose(
    translation: np.ndarray | None = None,
    rotation_y: float = 0.0,
    frame_id: int = 0,
) -> Pose:
    """构造 pose (绕 y 轴旋转)."""
    if translation is None:
        translation = np.zeros(3, dtype=np.float64)
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    m[:3, 3] = translation
    return Pose(matrix=m, frame_id=frame_id)


def _make_object(
    object_id: str = "chair_000_00",
    label: str = "chair",
    center: np.ndarray | None = None,
    confidence: float = 0.9,
) -> Object3D:
    if center is None:
        center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return Object3D(object_id=object_id, label=label, center=center, confidence=confidence)


class TestCoordinateTransform:
    """AC: 坐标变换双向可验证."""

    def test_world_to_camera_basic(self) -> None:
        p_world = np.array([1.0, 0.0, 0.0])
        pose = _make_pose(translation=np.array([5.0, 0.0, 0.0]))
        p_cam = WorldModel.world_to_camera(p_world, pose)
        assert p_cam.shape == (3,)

    def test_camera_to_world_basic(self) -> None:
        p_cam = np.array([1.0, 0.0, 0.0])
        pose = _make_pose(translation=np.array([5.0, 0.0, 0.0]))
        p_world = WorldModel.camera_to_world(p_cam, pose)
        assert p_world.shape == (3,)

    def test_roundtrip_identity(self) -> None:
        """AC: roundtrip (identity pose) → 原坐标."""
        p_world = np.array([3.0, 4.0, 5.0])
        pose = Pose.identity()
        p_cam = WorldModel.world_to_camera(p_world, pose)
        p_back = WorldModel.camera_to_world(p_cam, pose)
        assert np.allclose(p_world, p_back, atol=1e-10)

    def test_roundtrip_translation(self) -> None:
        """AC: roundtrip (平移 pose) → 原坐标."""
        p_world = np.array([3.0, 4.0, 5.0])
        pose = _make_pose(translation=np.array([10.0, -5.0, 2.0]))
        p_cam = WorldModel.world_to_camera(p_world, pose)
        p_back = WorldModel.camera_to_world(p_cam, pose)
        assert np.allclose(p_world, p_back, atol=1e-10)

    def test_roundtrip_rotation(self) -> None:
        """AC: roundtrip (旋转 pose) → 原坐标."""
        p_world = np.array([3.0, 4.0, 5.0])
        pose = _make_pose(rotation_y=0.7)  # ~40 度.
        p_cam = WorldModel.world_to_camera(p_world, pose)
        p_back = WorldModel.camera_to_world(p_cam, pose)
        assert np.allclose(p_world, p_back, atol=1e-10)

    def test_roundtrip_combined(self) -> None:
        """AC: roundtrip (旋转+平移) → 原坐标."""
        p_world = np.array([3.0, 4.0, 5.0])
        pose = _make_pose(translation=np.array([10, -5, 2]), rotation_y=0.7)
        p_cam = WorldModel.world_to_camera(p_world, pose)
        p_back = WorldModel.camera_to_world(p_cam, pose)
        assert np.allclose(p_world, p_back, atol=1e-10)

    def test_batch_transform(self) -> None:
        """批量坐标变换."""
        pts_world = np.array([[1, 0, 0], [2, 3, 4], [5, 6, 7]], dtype=np.float64)
        pose = _make_pose(translation=np.array([10, 0, 0]))
        pts_cam = WorldModel.world_to_camera(pts_world, pose)
        assert pts_cam.shape == (3, 3)
        pts_back = WorldModel.camera_to_world(pts_cam, pose)
        assert np.allclose(pts_world, pts_back, atol=1e-10)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="p_world"):
            WorldModel.world_to_camera(np.array([1, 2]), Pose.identity())


class TestObjectManagement:
    def test_add_objects(self) -> None:
        obj = _make_object()
        ol = Object3DList(objects=[obj], frame_id=0)
        model = WorldModel()
        model.add_objects(ol)
        assert model.n_objects == 1

    def test_get_object(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("chair_000_00"))
        assert model.get_object("chair_000_00") is not None
        assert model.get_object("unknown") is None

    def test_object_pose(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a"))
        obj = model.object_pose("a")
        assert obj is not None
        assert obj.label == "chair"

    def test_object_in_world(self) -> None:
        center = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        model = WorldModel()
        model.add_object(_make_object("a", center=center))
        pos = model.object_in_world("a")
        assert np.allclose(pos, center)

    def test_object_in_camera(self) -> None:
        center = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        model = WorldModel()
        model.add_object(_make_object("a", center=center))
        # 无 pose → world == camera.
        pos_cam = model.object_in_camera("a")
        assert np.allclose(pos_cam, center)
        # 有 pose.
        pose = _make_pose(translation=np.array([10, 0, 0]))
        pos_cam = model.object_in_camera("a", pose)
        assert not np.allclose(pos_cam, center)


class TestObjectRelations:
    def test_object_distance(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([0, 0, 0], dtype=np.float32)))
        model.add_object(_make_object("b", center=np.array([3, 4, 0], dtype=np.float32)))
        d = model.object_distance("a", "b")
        assert d == 5.0  # 3-4-5 直角三角形.

    def test_object_distance_missing(self) -> None:
        model = WorldModel()
        d = model.object_distance("a", "b")
        assert d is None

    def test_object_direction(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([0, 0, 0], dtype=np.float32)))
        model.add_object(_make_object("b", center=np.array([3, 0, 0], dtype=np.float32)))
        direction = model.object_direction("a", "b")
        assert direction is not None
        assert np.allclose(direction, [1, 0, 0])
        # 单位向量.
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_object_direction_same_pos(self) -> None:
        model = WorldModel()
        center = np.array([1, 1, 1], dtype=np.float32)
        model.add_object(_make_object("a", center=center))
        model.add_object(_make_object("b", center=center))
        direction = model.object_direction("a", "b")
        assert np.allclose(direction, [0, 0, 0])

    def test_distance_to_camera_no_pose(self) -> None:
        """无 pose → 相机在原点."""
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([3, 4, 0], dtype=np.float32)))
        d = model.distance_to_camera("a")
        assert d == 5.0

    def test_distance_to_camera_with_pose(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([0, 0, 0], dtype=np.float32)))
        # 相机在 (10, 0, 0).
        pose = _make_pose(translation=np.array([10, 0, 0]))
        # 相机位置 = -R^T @ t, identity R → position = -t = (-10, 0, 0).
        # 但 translation = [10,0,0], R=I → position = -[10,0,0] = [-10,0,0]
        d = model.distance_to_camera("a", pose)
        assert d is not None
        assert d == 10.0  # 物体在原点, 相机在 -10.

    def test_direction_from_camera(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([5, 0, 0], dtype=np.float32)))
        direction = model.direction_from_camera("a")  # 无 pose → 相机在原点.
        assert direction is not None
        assert np.allclose(direction, [1, 0, 0])


class TestFindNearest:
    def test_find_nearest(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([0, 0, 0], dtype=np.float32)))
        model.add_object(_make_object("b", center=np.array([1, 0, 0], dtype=np.float32)))
        model.add_object(_make_object("c", center=np.array([5, 0, 0], dtype=np.float32)))
        result = model.find_nearest("a")
        assert result is not None
        nearest_id, dist = result
        assert nearest_id == "b"
        assert dist == 1.0

    def test_find_nearest_with_label(self) -> None:
        model = WorldModel()
        model.add_object(
            _make_object("a", label="chair", center=np.array([0, 0, 0], dtype=np.float32))
        )
        model.add_object(
            _make_object("b", label="table", center=np.array([1, 0, 0], dtype=np.float32))
        )
        model.add_object(
            _make_object("c", label="table", center=np.array([5, 0, 0], dtype=np.float32))
        )
        result = model.find_nearest("a", label="table")
        assert result is not None
        assert result[0] == "b"  # 最近的 table.


class TestSIUnits:
    """AC: 单位统一为 SI."""

    def test_distance_in_meters(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([0, 0, 0], dtype=np.float32)))
        model.add_object(_make_object("b", center=np.array([10, 0, 0], dtype=np.float32)))
        d = model.object_distance("a", "b")
        assert d == 10.0  # 10 米.

    def test_coordinates_in_meters(self) -> None:
        center = np.array([5.5, 3.2, 1.8], dtype=np.float32)
        model = WorldModel()
        model.add_object(_make_object("a", center=center))
        pos = model.object_in_world("a")
        # 坐标值 == 米数.
        assert np.allclose(pos, center)

    def test_direction_unit_vector(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", center=np.array([0, 0, 0], dtype=np.float32)))
        model.add_object(_make_object("b", center=np.array([3, 4, 0], dtype=np.float32)))
        direction = model.object_direction("a", "b")
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6  # 单位向量.


class TestSummary:
    def test_summary(self) -> None:
        model = WorldModel()
        model.add_object(_make_object("a", label="chair"))
        model.add_object(_make_object("b", label="table"))
        s = model.summary()
        assert s["n_objects"] == 2
        assert "chair" in s["labels"]
        assert s["unit_length"] == "meter"
        assert s["unit_angle"] == "radian"
