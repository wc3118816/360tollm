"""M038 LLM 空间问答 Agent 子包.

公开 API:
    SpatialQA       - 端到端问答 Agent (问题 → 答案 + 证据)
    QuestionParser  - 自然语言 → ContextQuery
    AnswerGenerator - ContextSnippet → Answer
    Answer          - 答案 (text + evidence + confidence)

AC: 答案能够引用 World Model 中的证据字段.
实现: Answer.evidence 列表直接引用 snippet.objects 中的字段
    (object_id / position / confidence / distance 等).

用法:
    from modules.spatial_qa import SpatialQA
    qa = SpatialQA(world_model=model, spatial_memory=mem)
    answer = qa.ask("椅子在哪里?")
    print(answer.text)       # "椅子在 (1.0, 2.0, 3.0) 米处..."
    print(answer.evidence)   # [{object_id, position, confidence}, ...]
"""

from __future__ import annotations

from .answer import Answer, AnswerGenerator
from .parser import QuestionParser
from .qa import SpatialQA

__all__ = [
    "SpatialQA",
    "QuestionParser",
    "AnswerGenerator",
    "Answer",
]
