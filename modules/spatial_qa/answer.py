"""M038 LLM 空间问答: 答案生成.

AnswerGenerator:
- generate(question, snippet) → Answer.
- 将 ContextSnippet (LLM 上下文) 转为自然语言答案.
- AC: 答案引用 World Model 中的证据字段.

Answer:
- text: str 自然语言答案.
- evidence: list[dict] 证据字段 (来自 World Model).
- confidence: float 答案置信度.
- query_type: str 查询类型.
- snippet: ContextSnippet 原始上下文.

用法:
    gen = AnswerGenerator()
    answer = gen.generate("椅子在哪里?", snippet)
    print(answer.text)        # "椅子在 (1.0, 2.0, 3.0) 米处..."
    print(answer.evidence)   # [{object_id, position, confidence}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.llm_context import ContextSnippet, QueryType
from modules.logging import get_logger

_LOG = get_logger("modules.spatial_qa.answer")


@dataclass
class Answer:
    """空间问答答案.

    Attributes:
        text: 自然语言答案.
        evidence: 证据字段 (来自 World Model).
        confidence: 答案置信度.
        query_type: 查询类型.
        snippet: 原始上下文.
    """

    text: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    query_type: str = ""
    snippet: ContextSnippet | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 4),
            "query_type": self.query_type,
        }


@dataclass
class AnswerGenerator:
    """答案生成器.

    将 ContextSnippet 转为自然语言答案 + 证据字段.

    AC: 答案能够引用 World Model 中的证据字段.
    实现: evidence 列表直接引用 snippet.objects 中的字段
        (object_id / position / confidence / distance 等).
    """

    def __post_init__(self) -> None:
        self._log = _LOG

    def generate(
        self,
        question: str,
        snippet: ContextSnippet,
        query_type: QueryType = QueryType.CUSTOM,
    ) -> Answer:
        """生成答案.

        Args:
            question: 原始问题.
            snippet: LLM 上下文片段.
            query_type: 查询类型.

        Returns:
            Answer 答案 + 证据.
        """
        # 答案文本 = snippet.text (已由 LLMContextAdapter 生成).
        text = snippet.text

        # 证据字段 = snippet.objects (直接引用 World Model 字段).
        evidence = snippet.objects.copy()

        # 答案置信度 = snippet.confidence.
        confidence = snippet.confidence

        answer = Answer(
            text=text,
            evidence=evidence,
            confidence=confidence,
            query_type=query_type.value,
            snippet=snippet,
        )
        self._log.debug(
            "answer generated",
            query_type=query_type.value,
            n_evidence=len(evidence),
            confidence=confidence,
        )
        return answer
