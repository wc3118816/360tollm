"""M021 单元测试: ObjectAssociator (2D Detection + 深度 + Pose → 3D).

覆盖:
- 基本关联 (equirect + perspective)
- 2D bbox → 3D center (世界坐标)
- 无效深度 → 跳过
- object_id 自动生成
- confidence 传递
- 3D 尺寸估算
- 多个 detection
- 坐标单位 (米)
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.depth.types import DepthMap
from modules.detection import Detection, DetectionList
from modules.objects import AssociatorConfig, ObjectAssociator
from modules.vo.types import Pose


def _make_depth_map(h: int = 100, w: int = 200, depth_val: float = 5.0) -> DepthMap:
    depth = np.full((h, w), depth_val, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    return DepthMap(depth=depth, confidence=conf)


def _make_detection(
    bbox: list[int] | None = None,
    label: str = "chair",
    confidence: float = 0.9,
) -> Detection:
    if bbox is None:
        bbox = [50, 40, 100, 80]
    return Detection(
        bbox=np.array(bbox, dtype=np.int32),
        label=label,
        label_id=0,
        confidence=confidence,
    )


class TestAssociation:
    def test_basic_equirect(self) -> None:
        dm = _make_depth_map(100, 200, depth_val=5.0)
        det = _make_detection()
        dl = DetectionList(
            detections=[det], latency_ms=1.0, backend="dummy", image_shape=(100, 200)
        )
        pose = Pose.identity()
        assoc = ObjectAssociator(AssociatorConfig(projection="equirect"))
        ol = assoc.associate(dl, dm, pose)
        assert ol.n_objects == 1
        assert ol.objects[0].label == "chair"
        assert ol.objects[0].confidence == 0.9

    def test_basic_perspective(self) -> None:
        dm = _make_depth_map(100, 200, depth_val=5.0)
        det = _make_detection()
        dl = DetectionList(detections=[det], image_shape=(100, 200))
        K = np.array([[200, 0, 100], [0, 200, 50], [0, 0, 1]], dtype=np.float64)
        pose = Pose.identity()
        assoc = ObjectAssociator(AssociatorConfig(projection="perspective"))
        ol = assoc.associate(dl, dm, pose, camera_matrix=K)
        assert ol.n_objects == 1

    def test_perspective_no_K_raises(self) -> None:
        dm = _make_depth_map()
        det = _make_detection()
        dl = DetectionList(detections=[det])
        pose = Pose.identity()
        assoc = ObjectAssociator(AssociatorConfig(projection="perspective"))
        with pytest.raises(ValueError, match="camera_matrix"):
            assoc.associate(dl, dm, pose)

    def test_invalid_depth_skipped(self) -> None:
        """无效深度 → 跳过."""
        depth = np.full((100, 200), 0.05, dtype=np.float32)  # < min_depth.
        conf = np.ones((100, 200), dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        det = _make_detection()
        dl = DetectionList(detections=[det])
        pose = Pose.identity()
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, pose)
        assert ol.n_objects == 0  # 被跳过.

    def test_object_id_generation(self) -> None:
        dm = _make_depth_map()
        det1 = _make_detection(label="chair")
        det2 = _make_detection(label="chair", bbox=[10, 10, 50, 40])
        dl = DetectionList(detections=[det1, det2])
        pose = Pose.identity(frame_id=3)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, pose)
        # object_id = "{label}_{frame:03d}_{count:02d}".
        assert ol.objects[0].object_id == "chair_003_00"
        assert ol.objects[1].object_id == "chair_003_01"

    def test_confidence_passed(self) -> None:
        dm = _make_depth_map()
        det = _make_detection(confidence=0.75)
        dl = DetectionList(detections=[det])
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        assert ol.objects[0].confidence == 0.75

    def test_multiple_detections(self) -> None:
        dm = _make_depth_map()
        dets = [
            _make_detection(bbox=[10, 10, 50, 40], label="chair"),
            _make_detection(bbox=[60, 60, 100, 80], label="person"),
            _make_detection(bbox=[110, 10, 150, 40], label="cup"),
        ]
        dl = DetectionList(detections=dets)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        assert ol.n_objects == 3
        assert len(ol.labels) == 3

    def test_coordinates_in_meters(self) -> None:
        """AC: 坐标单位统一 (米)."""
        dm = _make_depth_map(depth_val=5.0)
        det = _make_detection()
        dl = DetectionList(detections=[det])
        pose = Pose.identity()
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, pose)
        center = ol.objects[0].center
        # 深度 5 米 → center 距离原点 ~5 米.
        dist = float(np.linalg.norm(center))
        assert 1.0 < dist < 20.0  # 在合理米数范围.

    def test_size_3d_positive(self) -> None:
        dm = _make_depth_map(depth_val=5.0)
        det = _make_detection()
        dl = DetectionList(detections=[det])
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        size = ol.objects[0].size_3d
        assert np.all(size > 0)  # 尺寸 > 0.

    def test_bbox_3d_eight_vertices(self) -> None:
        dm = _make_depth_map()
        det = _make_detection()
        dl = DetectionList(detections=[det])
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        assert ol.objects[0].bbox_3d.shape == (8, 3)


class TestPoseTransform:
    def test_pose_affects_center(self) -> None:
        """不同 pose → 不同世界坐标."""
        dm = _make_depth_map()
        det = _make_detection()
        dl = DetectionList(detections=[det])
        assoc = ObjectAssociator()

        ol1 = assoc.associate(dl, dm, Pose.identity())
        # 平移 pose.
        m2 = np.eye(4, dtype=np.float64)
        m2[:3, 3] = [10, 0, 0]
        pose2 = Pose(matrix=m2, frame_id=1)
        ol2 = assoc.associate(dl, dm, pose2)
        # 中心应不同.
        assert not np.allclose(ol1.objects[0].center, ol2.objects[0].center)
