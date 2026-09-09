"""M038 单元测试: QuestionParser + AnswerGenerator.

覆盖:
- QuestionParser.parse 各类问题
- AnswerGenerator.generate 答案 + 证据
- AC: 答案引用证据字段
"""

from __future__ import annotations

from modules.llm_context import ContextSnippet, QueryType
from modules.spatial_qa import AnswerGenerator, QuestionParser


class TestQuestionParser:
    def test_where_is(self) -> None:
        parser = QuestionParser()
        q = parser.parse("椅子在哪里?")
        assert q.query_type == QueryType.WHERE_IS
        assert q.target_label == "chair"

    def test_where_is_short(self) -> None:
        parser = QuestionParser()
        q = parser.parse("门在哪?")
        assert q.query_type == QueryType.WHERE_IS
        assert q.target_label == "door"

    def test_distance(self) -> None:
        parser = QuestionParser()
        q = parser.parse("桌子离我多远?")
        assert q.query_type == QueryType.DISTANCE
        assert q.target_label == "table"

    def test_direction(self) -> None:
        parser = QuestionParser()
        q = parser.parse("椅子在我左边吗?")
        assert q.query_type == QueryType.DIRECTION

    def test_count(self) -> None:
        parser = QuestionParser()
        q = parser.parse("有几个椅子?")
        assert q.query_type == QueryType.COUNT
        assert q.target_label == "chair"

    def test_nearest(self) -> None:
        parser = QuestionParser()
        q = parser.parse("离我最近的椅子?")
        assert q.query_type == QueryType.NEAREST

    def test_recent(self) -> None:
        parser = QuestionParser()
        q = parser.parse("刚才那个人去哪了?")
        assert q.query_type == QueryType.RECENT
        assert q.target_label == "person"
        assert q.time_window == 10.0

    def test_list_objects(self) -> None:
        parser = QuestionParser()
        q = parser.parse("有哪些物体?")
        assert q.query_type == QueryType.LIST_OBJECTS

    def test_custom(self) -> None:
        parser = QuestionParser()
        q = parser.parse("场景怎么样?")
        assert q.query_type == QueryType.CUSTOM

    def test_no_label(self) -> None:
        parser = QuestionParser()
        q = parser.parse("有哪些物体?")
        assert q.target_label is None


class TestAnswerGenerator:
    def test_generate_basic(self) -> None:
        snippet = ContextSnippet(
            text="椅子在 (1.0, 2.0, 3.0) 米处",
            facts=["椅子在 (1.0, 2.0, 3.0) 米处"],
            objects=[
                {
                    "object_id": "chair_000",
                    "label": "chair",
                    "position": [1.0, 2.0, 3.0],
                    "confidence": 0.9,
                }
            ],
            confidence=0.9,
            n_objects_filtered=1,
        )
        gen = AnswerGenerator()
        answer = gen.generate("椅子在哪里?", snippet, QueryType.WHERE_IS)
        assert "椅子" in answer.text
        assert len(answer.evidence) == 1
        assert answer.evidence[0]["object_id"] == "chair_000"
        assert answer.confidence == 0.9
        assert answer.query_type == "where_is"

    def test_generate_empty_snippet(self) -> None:
        snippet = ContextSnippet(text="未找到物体。", confidence=0.0)
        gen = AnswerGenerator()
        answer = gen.generate("沙发在哪里?", snippet, QueryType.WHERE_IS)
        assert answer.text == "未找到物体。"
        assert len(answer.evidence) == 0

    def test_ac_evidence_fields(self) -> None:
        """AC: 答案引用 World Model 中的证据字段."""
        snippet = ContextSnippet(
            text="椅子在 (1.0, 2.0, 3.0) 米处",
            objects=[
                {
                    "object_id": "chair_000",
                    "label": "chair",
                    "position": [1.0, 2.0, 3.0],
                    "confidence": 0.9,
                }
            ],
            confidence=0.9,
        )
        gen = AnswerGenerator()
        answer = gen.generate("椅子在哪里?", snippet, QueryType.WHERE_IS)
        # 证据字段包含 World Model 的字段.
        ev = answer.evidence[0]
        assert "object_id" in ev
        assert "label" in ev
        assert "position" in ev
        assert "confidence" in ev

    def test_to_dict(self) -> None:
        snippet = ContextSnippet(text="test", confidence=0.5)
        gen = AnswerGenerator()
        answer = gen.generate("test", snippet, QueryType.CUSTOM)
        d = answer.to_dict()
        assert "text" in d
        assert "evidence" in d
        assert "confidence" in d
