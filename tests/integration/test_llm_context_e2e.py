"""M037 集成测试: Image → Depth+Detection → WorldModel → LLM 上下文 端到端.

验证:
1. 完整流程: Image → Depth + Detection → Object3D → WorldModel → LLMContextAdapter
2. AC: 同一问题不会无条件发送整张地图
3. 多种查询类型
4. 与 SpatialMemory 集成 (RECENT 查询)
"""

from __future__ import annotations

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.llm_context import (
    ContextQuery,
    LLMContextAdapter,
    QueryType,
)
from modules.objects import ObjectAssociator
from modules.spatial_memory import SpatialMemory
from modules.vo.types import Pose
from modules.world_model import WorldModel


class TestFullPipeline:
    def _build_world_model(self) -> WorldModel:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        assoc = ObjectAssociator()
        dm = depth_est.estimate(img)
        dl = det.detect(img)
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)
        return model

    def test_image_to_llm_context(self) -> None:
        """完整流程: Image → LLM 上下文."""
        model = self._build_world_model()
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="有哪些物体?",
            query_type=QueryType.LIST_OBJECTS,
            max_objects=5,
        )
        snippet = adapter.adapt(q, world_model=model)
        assert snippet.text != ""
        assert len(snippet.objects) <= 5

    def test_ac_no_full_map(self) -> None:
        """AC: 同一问题不会无条件发送整张地图."""
        # 构造大场景.
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        assoc = ObjectAssociator()
        model = WorldModel()
        for _ in range(10):
            dm = depth_est.estimate(img)
            dl = det.detect(img)
            ol = assoc.associate(dl, dm, Pose.identity())
            model.add_objects(ol)

        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="有哪些物体?",
            query_type=QueryType.LIST_OBJECTS,
            max_objects=5,
        )
        snippet = adapter.adapt(q, world_model=model)
        # 不超过 max_objects.
        assert len(snippet.objects) <= 5
        # 但总数记录.
        assert snippet.n_objects_total >= 5

    def test_where_is_query(self) -> None:
        model = self._build_world_model()
        adapter = LLMContextAdapter()
        # 获取第一个物体的 label.
        if model.n_objects > 0:
            first_label = list(model.objects.values())[0].label
            q = ContextQuery(
                question=f"{first_label}在哪里?",
                query_type=QueryType.WHERE_IS,
                target_label=first_label,
                max_objects=3,
            )
            snippet = adapter.adapt(q, world_model=model)
            assert snippet.text != ""

    def test_count_query(self) -> None:
        model = self._build_world_model()
        adapter = LLMContextAdapter()
        q = ContextQuery(question="有几个物体?", query_type=QueryType.COUNT)
        snippet = adapter.adapt(q, world_model=model)
        assert str(model.n_objects) in snippet.text

    def test_nearest_query(self) -> None:
        model = self._build_world_model()
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="离我最近的物体?",
            query_type=QueryType.NEAREST,
            reference_position=np.array([0, 0, 0]),
        )
        snippet = adapter.adapt(q, world_model=model)
        assert snippet.text != ""

    def test_recent_with_memory(self) -> None:
        """RECENT 查询从 SpatialMemory."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=2, latency_ms=0))
        assoc = ObjectAssociator()
        mem = SpatialMemory()
        for i in range(3):
            dm = depth_est.estimate(img)
            dl = det.detect(img, timestamp=float(i))
            ol = assoc.associate(dl, dm, Pose.identity(frame_id=i))
            mem.observe(ol)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="刚才有什么物体?",
            query_type=QueryType.RECENT,
            time_window=10.0,
        )
        snippet = adapter.adapt(q, spatial_memory=mem)
        assert snippet.text != ""

    def test_custom_summary(self) -> None:
        model = self._build_world_model()
        adapter = LLMContextAdapter()
        q = ContextQuery(question="场景概况", query_type=QueryType.CUSTOM)
        snippet = adapter.adapt(q, world_model=model)
        assert str(model.n_objects) in snippet.text
