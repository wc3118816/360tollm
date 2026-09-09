"""M037 单元测试: LLMContextAdapter.

覆盖:
- ContextQuery 构造 + 校验
- ContextSnippet 构造 + to_dict
- WHERE_IS 查询
- DISTANCE 查询
- DIRECTION 查询
- COUNT 查询
- LIST_OBJECTS 查询 + max_objects 限制 (AC: 不发送整张地图)
- NEAREST 查询
- RECENT 查询 (从 SpatialMemory)
- CUSTOM 查询
- AC: 同一问题不无条件发送整张地图
- 空模型
- 方向描述
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.llm_context import (
    ContextQuery,
    ContextSnippet,
    LLMContextAdapter,
    QueryType,
)
from modules.objects import Object3D, Object3DList
from modules.spatial_memory import SpatialMemory
from modules.world_model import WorldModel


def _make_world_model(n_chairs: int = 3, n_tables: int = 2) -> WorldModel:
    """构造测试用 WorldModel."""
    model = WorldModel()
    objs = []
    for i in range(n_chairs):
        objs.append(
            Object3D(
                object_id=f"chair_{i:03d}",
                label="chair",
                center=np.array([float(i), 0.0, 0.0], dtype=np.float32),
                confidence=min(0.8 + 0.02 * i, 0.99),
            )
        )
    for i in range(n_tables):
        objs.append(
            Object3D(
                object_id=f"table_{i:03d}",
                label="table",
                center=np.array([10.0 + float(i), 0.0, 0.0], dtype=np.float32),
                confidence=0.7,
            )
        )
    model.add_objects(Object3DList(objects=objs))
    return model


class TestContextQuery:
    def test_basic_construction(self) -> None:
        q = ContextQuery(question="椅子在哪里?", query_type=QueryType.WHERE_IS)
        assert q.question == "椅子在哪里?"
        assert q.query_type == QueryType.WHERE_IS

    def test_max_objects_default(self) -> None:
        q = ContextQuery(question="test")
        assert q.max_objects == 10

    def test_max_objects_minimum(self) -> None:
        q = ContextQuery(question="test", max_objects=0)
        assert q.max_objects == 1  # 最小 1.

    def test_reference_position_shape(self) -> None:
        with pytest.raises(ValueError, match="reference_position"):
            ContextQuery(question="test", reference_position=np.array([1, 2]))


class TestContextSnippet:
    def test_construction(self) -> None:
        s = ContextSnippet(text="test", facts=["a"], confidence=0.9)
        assert s.text == "test"
        assert s.facts == ["a"]
        assert s.confidence == 0.9

    def test_to_dict(self) -> None:
        s = ContextSnippet(text="test", n_objects_filtered=3, n_objects_total=10)
        d = s.to_dict()
        assert d["text"] == "test"
        assert d["n_objects_filtered"] == 3


class TestWhereIs:
    def test_where_is_chair(self) -> None:
        model = _make_world_model()
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="椅子在哪里?",
            query_type=QueryType.WHERE_IS,
            target_label="chair",
            max_objects=3,
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "chair" in snippet.text
        assert len(snippet.objects) == 3
        assert snippet.n_objects_filtered == 3

    def test_where_is_not_found(self) -> None:
        model = _make_world_model()
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="沙发在哪里?",
            query_type=QueryType.WHERE_IS,
            target_label="sofa",
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "未找到" in snippet.text


class TestDistance:
    def test_distance_with_reference(self) -> None:
        model = _make_world_model(n_chairs=1)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="椅子离我多远?",
            query_type=QueryType.DISTANCE,
            target_label="chair",
            reference_position=np.array([0, 0, 0]),
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "米" in snippet.text

    def test_distance_between_objects(self) -> None:
        model = _make_world_model(n_chairs=2)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="两张椅子距离?",
            query_type=QueryType.DISTANCE,
            target_object_id="chair_000:chair_001",
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "米" in snippet.text


class TestDirection:
    def test_direction_with_reference(self) -> None:
        model = _make_world_model(n_chairs=1)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="椅子在我哪个方向?",
            query_type=QueryType.DIRECTION,
            target_label="chair",
            reference_position=np.array([0, 0, 0]),
        )
        snippet = adapter.adapt(q, world_model=model)
        assert snippet.text != ""


class TestCount:
    def test_count_chairs(self) -> None:
        model = _make_world_model(n_chairs=3)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="有几个椅子?",
            query_type=QueryType.COUNT,
            target_label="chair",
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "3" in snippet.text

    def test_count_all(self) -> None:
        model = _make_world_model(n_chairs=3, n_tables=2)
        adapter = LLMContextAdapter()
        q = ContextQuery(question="有几个物体?", query_type=QueryType.COUNT)
        snippet = adapter.adapt(q, world_model=model)
        assert "5" in snippet.text


class TestListObjects:
    def test_list_all_objects(self) -> None:
        model = _make_world_model(n_chairs=3, n_tables=2)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="有哪些物体?",
            query_type=QueryType.LIST_OBJECTS,
            max_objects=10,
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "chair" in snippet.text
        assert "table" in snippet.text

    def test_ac_max_objects_limit(self) -> None:
        """AC: 同一问题不会无条件发送整张地图."""
        model = _make_world_model(n_chairs=20, n_tables=20)  # 40 个物体.
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="有哪些物体?",
            query_type=QueryType.LIST_OBJECTS,
            max_objects=5,  # 只返回 5 个.
        )
        snippet = adapter.adapt(q, world_model=model)
        assert len(snippet.objects) <= 5  # 不超过 max_objects.
        assert snippet.n_objects_total == 40  # 总数记录.


class TestNearest:
    def test_nearest_chair(self) -> None:
        model = _make_world_model(n_chairs=3)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="离我最近的椅子?",
            query_type=QueryType.NEAREST,
            target_label="chair",
            reference_position=np.array([0, 0, 0]),
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "最近" in snippet.text
        assert snippet.n_objects_filtered == 1  # 只返回 1 个.


class TestRecent:
    def test_recent_from_memory(self) -> None:
        mem = SpatialMemory()
        ol = Object3DList(
            objects=[
                Object3D(
                    object_id="obj_000_00",
                    label="chair",
                    center=np.array([1, 0, 0], dtype=np.float32),
                    confidence=0.9,
                )
            ],
            frame_id=0,
            timestamp=1.0,
        )
        mem.observe(ol)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="刚才那个物体?",
            query_type=QueryType.RECENT,
            time_window=10.0,
        )
        snippet = adapter.adapt(q, spatial_memory=mem)
        assert "chair" in snippet.text


class TestCustom:
    def test_custom_summary(self) -> None:
        model = _make_world_model(n_chairs=2, n_tables=0)
        adapter = LLMContextAdapter()
        q = ContextQuery(question="场景概况", query_type=QueryType.CUSTOM)
        snippet = adapter.adapt(q, world_model=model)
        assert "2" in snippet.text  # 2 个物体.


class TestACNoFullMap:
    """AC: 同一问题不会无条件发送整张地图."""

    def test_where_is_limited(self) -> None:
        """WHERE_IS 只返回 max_objects 个物体."""
        model = _make_world_model(n_chairs=50, n_tables=0)  # 50 个椅子.
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="椅子在哪里?",
            query_type=QueryType.WHERE_IS,
            target_label="chair",
            max_objects=5,
        )
        snippet = adapter.adapt(q, world_model=model)
        assert len(snippet.objects) <= 5  # 不超过 5 个.
        assert snippet.n_objects_total == 50  # 但总数记录.

    def test_nearest_single_object(self) -> None:
        """NEAREST 只返回 1 个物体."""
        model = _make_world_model(n_chairs=50, n_tables=0)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="最近的椅子?",
            query_type=QueryType.NEAREST,
            target_label="chair",
        )
        snippet = adapter.adapt(q, world_model=model)
        assert snippet.n_objects_filtered == 1  # 只 1 个.

    def test_count_no_objects_returned(self) -> None:
        """COUNT 不返回物体详情."""
        model = _make_world_model(n_chairs=50, n_tables=0)
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="有几个椅子?",
            query_type=QueryType.COUNT,
            target_label="chair",
        )
        snippet = adapter.adapt(q, world_model=model)
        assert len(snippet.objects) == 0  # 不返回物体列表.
        assert snippet.n_objects_filtered == 0


class TestEmptyModel:
    def test_empty_where_is(self) -> None:
        model = WorldModel()
        adapter = LLMContextAdapter()
        q = ContextQuery(
            question="椅子在哪里?",
            query_type=QueryType.WHERE_IS,
            target_label="chair",
        )
        snippet = adapter.adapt(q, world_model=model)
        assert "未找到" in snippet.text

    def test_none_model(self) -> None:
        adapter = LLMContextAdapter()
        q = ContextQuery(question="test", query_type=QueryType.WHERE_IS)
        snippet = adapter.adapt(q)
        assert "无法" in snippet.text


class TestDirectionText:
    def test_direction_forward(self) -> None:
        adapter = LLMContextAdapter()
        text = adapter._direction_to_text(np.array([1, 0, 0]))
        assert "前方" in text

    def test_direction_right(self) -> None:
        adapter = LLMContextAdapter()
        text = adapter._direction_to_text(np.array([0, 0, 1]))
        assert "右侧" in text

    def test_direction_up(self) -> None:
        adapter = LLMContextAdapter()
        text = adapter._direction_to_text(np.array([0, 1, 0]))
        assert "上方" in text
