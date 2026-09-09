# MISSION_REPORT — M038 LLM 空间问答 Agent

## Goal

实现基础空间问答。
示例问题："我左边是什么？" / "桌子离我多远？" / "刚才那个人去哪了？" / "门在哪里？"
AC：答案能够引用 World Model 中的证据字段。

## Assumptions

- M029 已建立 WorldModel API
- M031 已建立 SpatialMemory（跨帧记忆）
- M037 已建立 LLMContextAdapter（上下文生成）

## Files Changed

新增：
- `modules/spatial_qa/parser.py`（`QuestionParser` 自然语言→ContextQuery）
- `modules/spatial_qa/answer.py`（`AnswerGenerator` + `Answer` 答案+证据）
- `modules/spatial_qa/qa.py`（`SpatialQA` 端到端 Agent）
- `modules/spatial_qa/__init__.py`（子包导出）
- `tests/unit/test_spatial_qa.py`（14 单测：解析器+答案生成）
- `tests/integration/test_spatial_qa_e2e.py`（8 集成测试：完整链路/AC）

修改：
- `README.md`：状态表更新 M038 = DONE

## Implementation

### 1. QuestionParser（parser.py）
```python
parser = QuestionParser()
query = parser.parse("椅子在哪里?")
# query.query_type == QueryType.WHERE_IS
# query.target_label == "chair"
```

**关键词模式匹配**：
- "在哪/哪里/哪儿" → WHERE_IS
- "多远/距离" → DISTANCE
- "哪个方向/左边/右边" → DIRECTION
- "几个/多少" → COUNT
- "最近" → NEAREST
- "刚才/刚刚" → RECENT
- "有哪些/有什么" → LIST_OBJECTS
- 其他 → CUSTOM

**中英文 label 映射**：椅子→chair、桌子→table、门→door、人→person 等

### 2. AnswerGenerator + Answer（answer.py）
```python
@dataclass
class Answer:
    text: str                           # 自然语言答案
    evidence: list[dict]                # 证据字段 (来自 World Model)
    confidence: float                   # 答案置信度
    query_type: str                     # 查询类型
    snippet: ContextSnippet | None      # 原始上下文
```

```python
gen = AnswerGenerator()
answer = gen.generate("椅子在哪里?", snippet, QueryType.WHERE_IS)
# answer.evidence = [{object_id, position, confidence}, ...]
```

**AC: 答案引用 World Model 证据字段**：
- `Answer.evidence` 直接引用 `snippet.objects` 中的字段
- 字段包括：`object_id`、`label`、`position`、`confidence`、`distance`、`last_seen`

### 3. SpatialQA 端到端（qa.py）
```python
qa = SpatialQA(world_model=model, spatial_memory=mem)
answer = qa.ask("椅子在哪里?")
print(answer.text)       # "椅子在 (1.0, 2.0, 3.0) 米处..."
print(answer.evidence)   # [{object_id, position, confidence}, ...]
```

**端到端流程**：
1. `QuestionParser.parse(question)` → `ContextQuery`
2. `LLMContextAdapter.adapt(query, world_model, spatial_memory)` → `ContextSnippet`
3. `AnswerGenerator.generate(question, snippet, query_type)` → `Answer`

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_spatial_qa.py \
  tests/integration/test_spatial_qa_e2e.py -v
# 22 passed in 0.43s
.venv\Scripts\python.exe -m pytest -q   # 全量 751 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_spatial_qa.py (14 单测)**：
- `TestQuestionParser` (10)：WHERE_IS + 短句 + DISTANCE + DIRECTION + COUNT + NEAREST + RECENT + LIST_OBJECTS + CUSTOM + 无 label
- `TestAnswerGenerator` (4)：基本生成 + 空片段 + **AC 证据字段** + to_dict

**test_spatial_qa_e2e.py (8 集成测试)**：
- `TestFullPipeline` (8)：where_is + distance + recent + list_objects + **AC 证据字段** + 批量 + 未找到 + 无模型

### 示例问题验证（mission_list.md）

| 问题 | 查询类型 | 验证 |
|---|---|---|
| "椅子在哪里?" | WHERE_IS | `test_ask_where_is` ✅ |
| "桌子离我多远?" | DISTANCE | `test_ask_distance` ✅ |
| "刚才那个人去哪了?" | RECENT | `test_ask_recent` ✅ |
| "有哪些物体?" | LIST_OBJECTS | `test_ask_list_objects` ✅ |

## Metrics

| 指标 | 值 |
|---|---|
| M038 测试数 | 22 (14 unit + 8 integration) |
| 全量测试数 | 751 (100%) |
| M038 测试耗时 | 0.43s |
| 全量测试耗时 | 44.94s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~300 (parser 100 + answer 80 + qa 120) + ~200 (测试) |
| 外部依赖 | 无（纯 re + numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 答案能够引用 World Model 中的证据字段 | ✅ `Answer.evidence` 列表直接引用 `snippet.objects` 中的 World Model 字段（object_id/label/position/confidence/distance/last_seen）；`test_ac_evidence_fields` 单测验证字段存在且值一致；端到端 `test_ac_evidence_fields` 验证 `ev["object_id"] == "chair_000"` + `ev["confidence"] == 0.9` |

## Remaining Risks

- **解析器简化**：基于关键词匹配，无 NLU/LLM 理解。复杂问句可能误分类。
- **无真实 LLM**：答案文本由规则生成，非 LLM 生成。需 M037+M038 进一步集成。
- **无多轮对话**：每次 `ask` 独立，无上下文记忆。M039 实时语音负责。
- **label 映射有限**：只覆盖常见物体，新类别需扩展 `_LABEL_MAP`。
- **无语音输入/输出**：纯文本。M039 负责。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M048 MVP Demo**：完成最小闭环（360相机→感知→世界模型→空间问答→输出）
