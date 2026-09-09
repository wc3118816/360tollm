"""M038 LLM 空间问答: 端到端 SpatialQA Agent.

SpatialQA:
- ask(question) → Answer.
- 端到端: 问题 → QuestionParser → LLMContextAdapter → AnswerGenerator → Answer.
- AC: 答案能够引用 World Model 中的证据字段.

用法:
    qa = SpatialQA(world_model=model, spatial_memory=mem)
    answer = qa.ask("椅子在哪里?")
    print(answer.text)       # 自然语言答案
    print(answer.evidence)   # 证据字段
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from modules.llm_context import LLMContextAdapter, QueryType
from modules.logging import get_logger
from modules.spatial_memory import SpatialMemory
from modules.world_model import WorldModel

from .answer import Answer, AnswerGenerator
from .parser import QuestionParser

_LOG = get_logger("modules.spatial_qa")

_NDIM_3D = 3


@dataclass
class SpatialQA:
    """空间问答 Agent.

    端到端: 自然语言问题 → 答案 + 证据.

    Attributes:
        world_model: WorldModel (M029).
        spatial_memory: SpatialMemory (M031).
        parser: 问题解析器.
        adapter: LLM 上下文适配器.
        generator: 答案生成器.
    """

    world_model: WorldModel | None = None
    spatial_memory: SpatialMemory | None = None
    parser: QuestionParser = field(default_factory=QuestionParser)
    adapter: LLMContextAdapter = field(default_factory=LLMContextAdapter)
    generator: AnswerGenerator = field(default_factory=AnswerGenerator)

    def __post_init__(self) -> None:
        self._log = _LOG

    def ask(
        self,
        question: str,
        reference_position: np.ndarray | None = None,
    ) -> Answer:
        """端到端空间问答.

        Args:
            question: 自然语言问题.
            reference_position: 参考位置 (如 "我" 的位置, None 用原点).

        Returns:
            Answer 答案 + 证据字段.
        """
        # 1. 解析问题 → ContextQuery.
        query = self.parser.parse(question)

        # 注入参考位置 (如有).
        if (
            reference_position is not None
            and query.reference_position is None
            and query.query_type in (QueryType.DISTANCE, QueryType.DIRECTION, QueryType.NEAREST)
        ):
            query.reference_position = np.asarray(reference_position, dtype=np.float64).ravel()

        # 2. 生成 LLM 上下文.
        snippet = self.adapter.adapt(
            query,
            world_model=self.world_model,
            spatial_memory=self.spatial_memory,
        )

        # 3. 生成答案 + 证据.
        answer = self.generator.generate(question, snippet, query.query_type)

        self._log.info(
            "ask",
            question=question,
            query_type=query.query_type.value,
            confidence=answer.confidence,
            n_evidence=len(answer.evidence),
        )
        return answer

    def ask_batch(
        self,
        questions: list[str],
        reference_position: np.ndarray | None = None,
    ) -> list[Answer]:
        """批量问答."""
        return [self.ask(q, reference_position) for q in questions]

    def set_world_model(self, model: WorldModel) -> None:
        """更新 WorldModel."""
        self.world_model = model

    def set_spatial_memory(self, mem: SpatialMemory) -> None:
        """更新 SpatialMemory."""
        self.spatial_memory = mem
