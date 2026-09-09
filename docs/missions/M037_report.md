# MISSION_REPORT — M037 LLM 上下文适配器

## Goal

将 World Model 转换成适合 LLM 的最小必要上下文。
原则：不发送无关点云、几何事实优先、保留置信度、保留时间信息、支持按任务检索上下文。
AC：同一问题不会无条件发送整张地图。

## Assumptions

- M029 已建立 WorldModel API
- M031 已建立 SpatialMemory（跨帧记忆）
- M036 已建立 WorldModelSerializer（JSON Schema）

## Files Changed

新增：
- `modules/llm_context/types.py`（`ContextQuery` + `ContextSnippet` + `QueryType`）
- `modules/llm_context/adapter.py`（`LLMContextAdapter`）
- `modules/llm_context/__init__.py`（子包导出）
- `tests/unit/test_llm_context.py`（26 单测：查询类型/AC/空模型/方向）
- `tests/integration/test_llm_context_e2e.py`（7 集成测试：完整链路/AC）

修改：
- `README.md`：状态表更新 M037 = DONE

## Implementation

### 1. ContextQuery + ContextSnippet（types.py）
```python
class QueryType(StrEnum):
    WHERE_IS = "where_is"      # "X 在哪里?"
    DISTANCE = "distance"      # "X 离 Y 多远?"
    DIRECTION = "direction"   # "X 在 Y 的哪个方向?"
    COUNT = "count"            # "有几个 X?"
    LIST_OBJECTS = "list_objects"  # "有哪些物体?"
    NEAREST = "nearest"        # "离我最近的 X?"
    RECENT = "recent"          # "刚才那个物体?"
    CUSTOM = "custom"
```

```python
@dataclass
class ContextQuery:
    question: str
    query_type: QueryType
    target_label: str | None       # 目标类别
    target_object_id: str | None  # 目标物体 ID
    reference_position: np.ndarray | None  # 参考位置
    time_window: float | None     # 时间窗口
    max_objects: int = 10         # AC: 防止发送整张地图
```

### 2. LLMContextAdapter（adapter.py）
```python
adapter = LLMContextAdapter()
snippet = adapter.adapt(query, world_model, spatial_memory)
print(snippet.text)  # LLM 可读文本
```

**支持的查询类型**：
- `WHERE_IS` → 位置事实（"椅子在 (1.0, 2.0, 3.0) 米处"）
- `DISTANCE` → 距离事实（"距参考位置 5.0 米"）
- `DIRECTION` → 方向事实（"在参考位置的前方右侧"）
- `COUNT` → 数量事实（"有 3 个椅子"）
- `LIST_OBJECTS` → 物体列表（受 `max_objects` 限制）
- `NEAREST` → 最近物体（只返回 1 个）
- `RECENT` → 从 SpatialMemory 查询时间窗口内物体
- `CUSTOM` → 场景摘要

**原则对照**：

| 原则 | 实现 |
|---|---|
| 不发送无关点云 | 只返回物体摘要（ID/label/position/confidence），无原始点云 |
| 几何事实优先 | `facts` 列表含位置/距离/方向等自然语言事实 |
| 保留置信度 | 每个 `ContextSnippet.objects` 含 `confidence` 字段 |
| 保留时间信息 | `RECENT` 查询含 `last_seen` 时间戳 |
| 支持按任务检索 | 8 种 `QueryType` + label/object_id/position 过滤 |

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_llm_context.py \
  tests/integration/test_llm_context_e2e.py -v
# 33 passed in 0.54s
.venv\Scripts\python.exe -m pytest -q   # 全量 729 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_llm_context.py (26 单测)**：
- `TestContextQuery` (4)：构造 + 默认值 + 最小值 + 位置 shape
- `TestContextSnippet` (2)：构造 + to_dict
- `TestWhereIs` (2)：找到椅子 + 未找到
- `TestDistance` (2)：参考位置距离 + 两物体间距离
- `TestDirection` (1)：参考位置方向
- `TestCount` (2)：按类别 + 全部
- `TestListObjects` (2)：全部 + **AC max_objects 限制**
- `TestNearest` (1)：最近椅子
- `TestRecent` (1)：从记忆查询
- `TestCustom` (1)：场景摘要
- `TestACNoFullMap` (3)：**WHERE_IS 限制** + **NEAREST 单个** + **COUNT 无详情**
- `TestEmptyModel` (2)：空模型 + None 模型
- `TestDirectionText` (3)：前方 + 右侧 + 上方

**test_llm_context_e2e.py (7 集成测试)**：
- `TestFullPipeline` (7)：完整流程 + **AC 不发送整张地图** + where_is + count + nearest + recent + custom

## Metrics

| 指标 | 值 |
|---|---|
| M037 测试数 | 33 (26 unit + 7 integration) |
| 全量测试数 | 729 (100%) |
| M037 测试耗时 | 0.54s |
| 全量测试耗时 | 38.62s |
| Lint | All checks passed |
| Format | 5 files already formatted |
| 代码行数 | ~420 (types 100 + adapter 320) + ~330 (测试) |
| 外部依赖 | 无（纯 numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 同一问题不会无条件发送整张地图 | ✅ `max_objects` 限制（默认 10）；WHERE_IS/LIST_OBJECTS 限制返回数；NEAREST 只返回 1 个；COUNT 不返回物体详情；`test_ac_max_objects_limit`（40 物体→只 5 个）+ `test_where_is_limited`（50 物体→只 5 个）+ `test_nearest_single_object`（只 1 个）+ `test_count_no_objects_returned`（0 个详情）+ `test_ac_no_full_map`（端到端验证）全部通过 |

## Remaining Risks

- **无自然语言解析**：当前需手动构造 `ContextQuery`，未实现从自然语言问题自动解析。M038 空间问答负责。
- **方向描述简化**：只有 6 方向（前/后/上/下/左/右），无角度描述。
- **无 LLM 集成**：只生成上下文文本，未与实际 LLM API 对接。M038 负责。
- **无缓存**：每次查询都重新计算。可加缓存。
- **SpatialMemory 集成有限**：只有 RECENT 查询用 SpatialMemory，其他查询用 WorldModel。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M038 LLM 空间问答 Agent**：实现基础空间问答（自然语言 → ContextQuery → LLM 回答）
- **M048 MVP Demo**：完成最小闭环
