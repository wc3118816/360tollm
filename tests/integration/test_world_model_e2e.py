"""M029 集成测试: Image → Depth+Detection → 3D Object → WorldModel 端到端.

验证:
1. 完整流程: Image → Depth + Detection → Object3D → WorldModel
2. AC: 坐标变换双向可验证
3. AC: 单位统一为 SI (米)
4. 物体查询 (distance/direction)
5. 多帧更新
"""

from __future__ import annotations

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.objects import ObjectAssociator
from modules.vo.types import Pose
from modules.world_model import WorldModel


class TestFullPipeline:
    def test_image_to_world_model(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)
        assert model.n_objects == ol.n_objects

    def test_ac_roundtrip(self) -> None:
        """AC: 坐标变换双向可验证."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)

        # 取一个物体的世界坐标 → 变换 → 反变换.
        if model.n_objects > 0:
            obj_id = list(model.objects.keys())[0]
            p_world = model.object_in_world(obj_id)
            assert p_world is not None
            # 平移 pose.
            pose = Pose(
                matrix=np.array(
                    [
                        [1, 0, 0, 5],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float64,
                )
            )
            p_cam = WorldModel.world_to_camera(p_world, pose)
            p_back = WorldModel.camera_to_world(p_cam, pose)
            assert np.allclose(p_world, p_back, atol=1e-8)

    def test_ac_si_units(self) -> None:
        """AC: 单位统一为 SI (米)."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)

        # 所有物体坐标应在合理米数范围.
        for obj in model.objects.values():
            d = float(np.linalg.norm(obj.center))
            assert 0.1 < d < 100.0  # 0.1-100 米.

    def test_object_queries(self) -> None:
        """物体查询: distance + direction."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)

        if model.n_objects >= 2:
            ids = list(model.objects.keys())
            d = model.object_distance(ids[0], ids[1])
            assert d is not None
            assert d > 0  # 距离 > 0.
            direction = model.object_direction(ids[0], ids[1])
            assert direction is not None
            assert abs(np.linalg.norm(direction) - 1.0) < 1e-6  # 单位向量.

    def test_find_nearest(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)

        if model.n_objects >= 2:
            ids = list(model.objects.keys())
            result = model.find_nearest(ids[0])
            assert result is not None
            assert result[1] > 0  # 距离 > 0.

    def test_multi_frame_update(self) -> None:
        """多帧更新 WorldModel."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        assoc = ObjectAssociator()
        model = WorldModel()

        for i in range(3):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            model.add_objects(ol)

        assert model.n_objects >= 2  # 累计物体.
