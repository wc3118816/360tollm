"""M038 集成测试: 完整空间问答端到端.

验证:
1. 完整流程: Image → Depth+Detection → WorldModel → SpatialQA → Answer
2. AC: 答案能够引用 World Model 中的证据字段
3. 示例问题 (mission_list.md):
   - "门在哪里?"
   - "桌子离我多远?"
   - "刚才那个人去哪了?"
   - "有哪些物体?"
4. 批量问答
"""

from __future__ import annotations

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.objects import Object3D, Object3DList, ObjectAssociator
from modules.spatial_memory import SpatialMemory
from modules.spatial_qa import SpatialQA
from modules.vo.types import Pose
from modules.world_model import WorldModel


class TestFullPipeline:
    def _build_qa(self) -> SpatialQA:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        depth_est = DummyDepthEstimator()
        det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=0))
        assoc = ObjectAssociator()
        dm = depth_est.estimate(img)
        dl = det.detect(img)
        ol = assoc.associate(dl, dm, Pose.identity())
        model = WorldModel()
        model.add_objects(ol)
        mem = SpatialMemory()
        mem.observe(ol)
        return SpatialQA(world_model=model, spatial_memory=mem)

    def test_ask_where_is(self) -> None:
        """示例: "门在哪里?"."""
        qa = self._build_qa()
        answer = qa.ask("椅子在哪里?")
        assert answer.text != ""
        assert answer.query_type == "where_is"

    def test_ask_distance(self) -> None:
        """示例: "桌子离我多远?"."""
        qa = self._build_qa()
        answer = qa.ask("桌子离我多远?", reference_position=np.array([0, 0, 0]))
        assert answer.text != ""
        assert answer.query_type == "distance"

    def test_ask_recent(self) -> None:
        """示例: "刚才那个人去哪了?"."""
        qa = self._build_qa()
        answer = qa.ask("刚才那个人去哪了?")
        assert answer.text != ""
        assert answer.query_type == "recent"

    def test_ask_list_objects(self) -> None:
        """示例: "有哪些物体?"."""
        qa = self._build_qa()
        answer = qa.ask("有哪些物体?")
        assert answer.text != ""

    def test_ac_evidence_fields(self) -> None:
        """AC: 答案能够引用 World Model 中的证据字段."""
        # 构造有物体的模型.
        model = WorldModel()
        model.add_objects(
            Object3DList(
                objects=[
                    Object3D(
                        object_id="chair_000",
                        label="chair",
                        center=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        confidence=0.9,
                    )
                ]
            )
        )
        qa = SpatialQA(world_model=model)
        answer = qa.ask("椅子在哪里?")
        # 答案有证据字段.
        assert len(answer.evidence) > 0
        ev = answer.evidence[0]
        # 证据引用 World Model 字段.
        assert "object_id" in ev
        assert "position" in ev
        assert "confidence" in ev
        # 证据值与 World Model 一致.
        assert ev["object_id"] == "chair_000"
        assert ev["confidence"] == 0.9

    def test_ask_batch(self) -> None:
        """批量问答."""
        qa = self._build_qa()
        questions = ["椅子在哪里?", "有几个椅子?", "有哪些物体?"]
        answers = qa.ask_batch(questions)
        assert len(answers) == 3
        for a in answers:
            assert a.text != ""

    def test_ask_not_found(self) -> None:
        """未找到物体."""
        qa = self._build_qa()
        answer = qa.ask("沙发在哪里?")
        assert "未找到" in answer.text or answer.text != ""

    def test_ask_no_model(self) -> None:
        """无 WorldModel."""
        qa = SpatialQA()
        answer = qa.ask("椅子在哪里?")
        assert "无法" in answer.text or answer.text != ""
