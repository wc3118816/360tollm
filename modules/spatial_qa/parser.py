"""M038 LLM 空间问答: 自然语言问题解析.

QuestionParser:
- parse(question) → ContextQuery.
- 基于关键词匹配的简单解析 (无 LLM 依赖).

支持的问句模式:
- "X 在哪里?" / "X 在哪?" / "哪里有 X?" → WHERE_IS
- "X 离 Y 多远?" / "X 离我多远?" → DISTANCE
- "X 在 Y 的哪个方向?" / "X 在哪边?" → DIRECTION
- "有几个 X?" / "有多少 X?" → COUNT
- "有哪些物体?" / "有什么?" → LIST_OBJECTS
- "离我最近的 X?" / "最近的 X 是?" → NEAREST
- "刚才那个 X?" / "最近有什么?" → RECENT
- 其他 → CUSTOM

用法:
    parser = QuestionParser()
    query = parser.parse("椅子在哪里?")
    # query.query_type == QueryType.WHERE_IS
    # query.target_label == "chair"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from modules.llm_context import ContextQuery, QueryType
from modules.logging import get_logger

_LOG = get_logger("modules.spatial_qa.parser")

# 中英文 label 映射 (常见物体).
_LABEL_MAP: dict[str, str] = {
    "椅子": "chair",
    "桌子": "table",
    "门": "door",
    "人": "person",
    "那个人": "person",
    "刚才那个人": "person",
    "沙发": "sofa",
    "床": "bed",
    "电视": "tv",
    "电脑": "computer",
    "书": "book",
    "杯子": "cup",
    "瓶": "bottle",
    "窗户": "window",
    "墙": "wall",
    "灯": "lamp",
    "植物": "plant",
    "花": "flower",
    "车": "car",
    " bicycle": "bicycle",
    "自行车": "bicycle",
}

# 关键词模式 (按优先级排序).
_PATTERNS: list[tuple[str, QueryType]] = [
    # RECENT: "刚才" / "刚刚" / "之前" — 优先级最高 (时间限定)
    (r"刚才|刚刚|之前", QueryType.RECENT),
    # DISTANCE: "多远" / "距离"
    (r"多远|距离|多长", QueryType.DISTANCE),
    # COUNT: "几个" / "多少"
    (r"几个|多少|多少个", QueryType.COUNT),
    # NEAREST: "最近" / "离我最近"
    (r"最近|离我最近", QueryType.NEAREST),
    # LIST_OBJECTS: "有哪些" / "有什么" / "是什么" — 优先于 DIRECTION
    (r"有哪些|有什么|什么东西|哪些物体|是什么|是谁", QueryType.LIST_OBJECTS),
    # WHERE_IS: "X 在哪里" / "X 在哪" / "哪儿"
    (r"在哪|在哪里|哪儿|什么地方", QueryType.WHERE_IS),
    # DIRECTION: "哪个方向" / "哪边" / "左边|右边|前面|后面" — 优先级最低
    (r"哪个方向|哪边|左边|右边|前方|后方|前面|后面|上面|下面", QueryType.DIRECTION),
]


@dataclass
class QuestionParser:
    """自然语言问题解析器.

    基于关键词匹配, 将问题字符串解析为 ContextQuery.
    """

    def __post_init__(self) -> None:
        self._log = _LOG

    def parse(self, question: str) -> ContextQuery:
        """解析自然语言问题为 ContextQuery.

        Args:
            question: 用户问题字符串.

        Returns:
            ContextQuery 查询对象.
        """
        q = question.strip()

        # 1. 匹配查询类型.
        query_type = self._match_query_type(q)

        # 2. 提取目标 label.
        target_label = self._extract_label(q)

        # 3. 提取目标物体 ID (如有 "A 和 B" / "A:B").
        target_object_id = self._extract_object_ids(q)

        # 4. 提取参考位置 (如有 "我" / "这里").
        reference_position = self._extract_reference(q)

        # 5. 提取时间窗口 (如有 "刚才").
        time_window = 10.0 if query_type == QueryType.RECENT else None

        query = ContextQuery(
            question=question,
            query_type=query_type,
            target_label=target_label,
            target_object_id=target_object_id,
            reference_position=reference_position,
            time_window=time_window,
        )
        self._log.debug(
            "parsed",
            question=question,
            query_type=query_type.value,
            label=target_label,
        )
        return query

    def _match_query_type(self, question: str) -> QueryType:
        """匹配查询类型."""
        for pattern, qtype in _PATTERNS:
            if re.search(pattern, question):
                return qtype
        return QueryType.CUSTOM

    def _extract_label(self, question: str) -> str | None:
        """提取目标类别."""
        for cn, en in _LABEL_MAP.items():
            if cn in question:
                return en
        return None

    def _extract_object_ids(self, question: str) -> str | None:
        """提取物体 ID (如有 'A 和 B' / 'A:B')."""
        # "A 和 B" 或 "A 与 B".
        m = re.search(r"(\w+)\s*[和与]\s*(\w+)", question)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        return None

    def _extract_reference(self, question: str) -> None:
        """提取参考位置.

        简化版: 如有 "我" / "这里" → 返回 None (表示用相机原点).
        实际位置由 SpatialQA 根据当前 pose 计算.
        """
        # 返回 None, 让 SpatialQA 用 current_pose 或原点.
        return None
