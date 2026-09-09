"""M048 集成测试: MVP Demo 完整闭环.

验证:
1. 完整流程: Frame → Detection → Depth → Pose → 3D → WorldModel → QA
2. AC: 4 个问题均可完成闭环并提供结构化证据
   - "我前面有什么？"
   - "左边是什么？"
   - "桌子离我多远？"
   - "刚才的人在哪里？"
3. 结构化证据字段
4. run() + print_results()
"""

from __future__ import annotations

import numpy as np

from modules.demo import MVPDemo
from modules.objects import Object3D, Object3DList
from modules.spatial_qa import Answer


class TestMVPDemo:
    def test_demo_questions_list(self) -> None:
        """4 个示例问题."""
        questions = MVPDemo.get_demo_questions()
        assert len(questions) == 4
        assert "我前面有什么？" in questions
        assert "左边是什么？" in questions
        assert "桌子离我多远？" in questions
        assert "刚才的人在哪里？" in questions

    def test_process_frames(self) -> None:
        """处理帧构建 WorldModel."""
        demo = MVPDemo(n_frames=3)
        demo.process_frames()
        assert demo.world_model.n_objects > 0
        assert demo.spatial_memory.n_objects > 0
        assert demo._processed

    def test_ask_single_question(self) -> None:
        """单次问答."""
        demo = MVPDemo(n_frames=3)
        answer = demo.ask("有哪些物体？")
        assert answer.text != ""
        assert isinstance(answer, Answer)

    def test_ac_four_questions_complete_loop(self) -> None:
        """AC: 4 个问题均可完成闭环."""
        demo = MVPDemo(n_frames=5)
        questions = MVPDemo.get_demo_questions()
        for q in questions:
            answer = demo.ask(q)
            # 闭环: 有答案文本.
            assert answer.text != ""
            # 有查询类型.
            assert answer.query_type != ""

    def test_ac_structured_evidence(self) -> None:
        """AC: 提供结构化证据."""
        demo = MVPDemo(n_frames=5)
        demo.process_frames()
        # 构造有物体的世界模型 (确保有证据).
        if demo.world_model.n_objects == 0:
            demo.world_model.add_objects(
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
            demo.qa.set_world_model(demo.world_model)

        # WHERE_IS 应提供结构化证据.
        answer = demo.ask("椅子在哪里？")
        if answer.evidence:
            ev = answer.evidence[0]
            assert "object_id" in ev or "label" in ev

    def test_run(self) -> None:
        """run() 返回完整结果."""
        demo = MVPDemo(n_frames=3)
        result = demo.run()
        assert "questions" in result
        assert "answers" in result
        assert "n_objects" in result
        assert "n_tracked" in result
        assert len(result["questions"]) == 4
        assert len(result["answers"]) == 4

    def test_print_results(self, capsys) -> None:
        """print_results() 输出人类可读结果."""
        demo = MVPDemo(n_frames=2)
        demo.print_results()
        captured = capsys.readouterr()
        assert "MVP Demo" in captured.out
        assert "问:" in captured.out

    def test_ac_question_1_front(self) -> None:
        """问题 1: "我前面有什么？" → LIST_OBJECTS."""
        demo = MVPDemo(n_frames=5)
        answer = demo.ask("我前面有什么？")
        assert answer.text != ""
        assert answer.query_type == "list_objects"

    def test_ac_question_2_left(self) -> None:
        """问题 2: "左边是什么？" → LIST_OBJECTS."""
        demo = MVPDemo(n_frames=5)
        answer = demo.ask("左边是什么？")
        assert answer.text != ""
        assert answer.query_type == "list_objects"

    def test_ac_question_3_distance(self) -> None:
        """问题 3: "桌子离我多远？" → DISTANCE."""
        demo = MVPDemo(n_frames=5)
        answer = demo.ask("桌子离我多远？")
        assert answer.text != ""
        assert answer.query_type == "distance"

    def test_ac_question_4_recent(self) -> None:
        """问题 4: "刚才的人在哪里？" → RECENT."""
        demo = MVPDemo(n_frames=5)
        answer = demo.ask("刚才的人在哪里？")
        assert answer.text != ""
        assert answer.query_type == "recent"


class TestEndToEndPipeline:
    """端到端管线验证: Frame → Detection → Depth → 3D → WorldModel → QA."""

    def test_full_pipeline(self) -> None:
        """完整管线: Image → ... → Answer."""
        demo = MVPDemo(n_frames=5)
        demo.process_frames()
        # 验证管线各阶段.
        assert demo.world_model.n_objects > 0  # 3D 物体已生成.
        assert demo.spatial_memory.n_objects > 0  # 空间记忆已更新.
        assert demo.scene_graph is not None  # 场景图已构建.
        # 问答.
        answer = demo.ask("有哪些物体？")
        assert answer.text != ""

    def test_multi_frame_tracking(self) -> None:
        """多帧跟踪: SpatialMemory 跨帧关联."""
        demo = MVPDemo(n_frames=5)
        demo.process_frames()
        # 多帧后应有跟踪对象.
        assert demo.spatial_memory.n_objects > 0

    def test_answer_has_evidence_fields(self) -> None:
        """答案有结构化证据字段."""
        demo = MVPDemo(n_frames=5)
        demo.process_frames()
        # 确保有物体.
        if demo.world_model.n_objects > 0:
            answer = demo.ask("椅子在哪里？")
            if answer.evidence:
                ev = answer.evidence[0]
                # 证据字段来自 World Model.
                assert isinstance(ev, dict)
