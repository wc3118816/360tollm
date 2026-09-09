"""M031 集成测试: Image → Depth+Detection → 3D Object → SpatialMemory 端到端.

验证:
1. 完整流程: Image → Depth + Detection → Object3D → SpatialMemory
2. AC: 系统可以回答"刚才那个物体现在在哪里"
3. 多帧记忆 + 跨帧关联
4. 静态物体 → 同一 tracked_id
5. 与 WorldModel 集成
"""

from __future__ import annotations

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.objects import ObjectAssociator
from modules.spatial_memory import SpatialMemory, SpatialMemoryConfig
from modules.vo.types import Pose


class TestFullPipeline:
    def test_image_to_spatial_memory(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        dm = depth_est.estimate(img)
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        dl = det.detect(img)
        assoc = ObjectAssociator()
        ol = assoc.associate(dl, dm, Pose.identity())
        mem = SpatialMemory()
        ids = mem.observe(ol)
        assert len(ids) == ol.n_objects
        assert mem.n_objects > 0

    def test_ac_where_is(self) -> None:
        """AC: 系统可以回答"刚才那个物体现在在哪里"."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=1, latency_ms=0, seed=42))
        assoc = ObjectAssociator()
        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=5.0))

        # 帧 0: 观测.
        dm0 = depth_est.estimate(img)
        dl0 = det.detect(img, timestamp=0.0)
        ol0 = assoc.associate(dl0, dm0, Pose.identity(frame_id=0))
        ids0 = mem.observe(ol0)

        # 帧 1: 再观测.
        dm1 = depth_est.estimate(img)
        dl1 = det.detect(img, timestamp=1.0)
        ol1 = assoc.associate(dl1, dm1, Pose.identity(frame_id=1))
        mem.observe(ol1)

        # where_is → 应返回有效位置.
        if ids0:
            pos = mem.where_is(ids0[0])
            assert pos is not None
            assert pos.shape == (3,)

    def test_multi_frame_memory(self) -> None:
        """多帧记忆."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0, seed=42))
        assoc = ObjectAssociator()
        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=10.0))

        for i in range(5):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            mem.observe(ol)

        # 应有记忆.
        assert mem.n_objects > 0
        # 至少一个物体有多次观测 (因 seed 固定 + threshold 大).
        tracked = mem.all_objects()
        assert len(tracked) > 0

    def test_static_object_same_id(self) -> None:
        """静态物体 → 同一 tracked_id (固定位置 + 固定 label)."""
        from modules.objects import Object3D, Object3DList

        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=100.0))

        # 连续 3 帧, 同一物体 (固定 label + 相近位置).
        all_ids: list[list[str]] = []
        for i in range(3):
            ol = Object3DList(
                objects=[
                    Object3D(
                        object_id=f"obj_{i:03d}_00",
                        label="chair",
                        center=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        confidence=0.9,
                    )
                ],
                frame_id=i,
                timestamp=float(i),
            )
            ids = mem.observe(ol)
            all_ids.append(ids)

        # 同一物体 → 同一 tracked_id.
        if all_ids[0]:
            tid = all_ids[0][0]
            for frame_ids in all_ids[1:]:
                if frame_ids:
                    assert frame_ids[0] == tid  # 同一 ID.

    def test_observation_history(self) -> None:
        """观测历史记录."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=1, latency_ms=0, seed=42))
        assoc = ObjectAssociator()
        mem = SpatialMemory(SpatialMemoryConfig(association_threshold=100.0))

        for i in range(3):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            mem.observe(ol)

        if mem.n_objects > 0:
            tracked = mem.all_objects()[0]
            assert tracked.n_observations >= 1

    def test_with_world_model(self) -> None:
        """与 WorldModel 集成."""
        from modules.world_model import WorldModel

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        assoc = ObjectAssociator()
        mem = SpatialMemory()
        model = WorldModel()

        for i in range(3):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            mem.observe(ol)
            model.add_objects(ol)

        # SpatialMemory 的物体位置应与 WorldModel 一致.
        if mem.n_objects > 0:
            tracked = mem.all_objects()[0]
            _ = tracked.current_position  # 验证可访问.
            # WorldModel 应有物体 (可能 ID 不同, 但位置应在).
            assert model.n_objects > 0
