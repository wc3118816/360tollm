"""M037 LLM 上下文适配器子包.

公开 API:
    LLMContextAdapter - World Model → LLM 最小必要上下文
    ContextQuery       - 查询定义 (question + query_type + filters)
    ContextSnippet     - 上下文片段 (text + facts + objects)
    QueryType          - 查询类型枚举

原则:
- 几何事实优先 (不发送原始点云)
- 保留置信度 + 时间信息
- max_objects 限制 → AC: 不无条件发送整张地图

用法:
    from modules.llm_context import LLMContextAdapter, ContextQuery, QueryType
    adapter = LLMContextAdapter()
    query = ContextQuery(question="椅子在哪里?", query_type=QueryType.WHERE_IS, target_label="chair")
    snippet = adapter.adapt(query, world_model, spatial_memory)
    print(snippet.text)
"""

from __future__ import annotations

from .adapter import LLMContextAdapter
from .types import ContextQuery, ContextSnippet, QueryType

__all__ = [
    "LLMContextAdapter",
    "ContextQuery",
    "ContextSnippet",
    "QueryType",
]
