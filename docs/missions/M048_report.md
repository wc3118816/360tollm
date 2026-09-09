# MISSION_REPORT — M048 MVP Demo

## Goal

完成最小闭环：
```
Osmo 360 → Video → Frame → Object Detection → Depth → Camera Pose
→ 3D Object → World Model → LLM → 用户问题
```

必须支持 4 个问题：
1. "我前面有什么？"
2. "左边是什么？"
3. "桌子离我多远？"
4. "刚才的人在哪里？"

AC：四个问题均可完成闭环并提供结构化证据。

## Assumptions

- M006 相机标定 ✓
- M009 深度估计 ✓
- M019 目标检测 ✓
- M021 2D→3D 关联 ✓
- M026 Scene Graph ✓
- M029 World Model API ✓
- M031 Spatial Memory ✓
- M036 World Model 序列化 ✓
- M037 LLM 上下文适配器 ✓
- M038 LLM 空间问答 ✓

## Files Changed

新增：
- `modules/demo/mvp_demo.py`（`MVPDemo` 端到端闭环）
- `modules/demo/__init__.py`（子包导出）
- `tests/integration/test_mvp_demo.py`（14 集成测试：4 问题/AC/管线）

修改：
- `modules/spatial_qa/parser.py`（修复 4 个问题解析优先级）
- `README.md`：状态表更新 M048 = DONE

## Implementation

### 1. QuestionParser 修复

**模式优先级调整**（支持 4 个示例问题）：

| 问题 | 旧优先级 | 新优先级 | 查询类型 |
|---|---|---|---|
| "我前面有什么？" | DIRECTION (先匹配"前面") | LIST_OBJECTS (先匹配"有什么") | list_objects |
| "左边是什么？" | DIRECTION (先匹配"左边") | LIST_OBJECTS (先匹配"是什么") | list_objects |
| "桌子离我多远？" | DISTANCE | DISTANCE (不变) | distance |
| "刚才的人在哪里？" | WHERE_IS (先匹配"在哪") | RECENT (先匹配"刚才") | recent |

**修复**：
- RECENT 提到最高优先级（时间限定优先）
- LIST_OBJECTS 提到 DIRECTION 之前（"有什么/是什么"优先于方向词）
- 添加"是什么|是谁"到 LIST_OBJECTS 模式

### 2. MVPDemo 端到端管线

```python
demo = MVPDemo(n_frames=5)
demo.run()           # 完整流程
demo.print_results()  # 打印结果
```

**完整管线**：
```
DummyFrame → DummyDetector → DummyDepthEstimator → IdentityPose
→ ObjectAssociator → WorldModel + SceneGraph + SpatialMemory
→ SpatialQA → Answer (text + evidence)
```

**MVP 使用 Dummy 后端**（无真实 360 视频/无真实 LLM）：
- `DummyDetector`: 合成检测结果
- `DummyDepthEstimator`: 合成深度图
- `Pose.identity()`: 相机在原点
- `ObjectAssociator`: 2D+Depth+Pose → 3D
- `SpatialQA`: 自然语言问答

### 3. AC 验证

**4 个问题闭环验证**：

| # | 问题 | 查询类型 | 测试 |
|---|---|---|---|
| 1 | "我前面有什么？" | list_objects | `test_ac_question_1_front` ✅ |
| 2 | "左边是什么？" | list_objects | `test_ac_question_2_left` ✅ |
| 3 | "桌子离我多远？" | distance | `test_ac_question_3_distance` ✅ |
| 4 | "刚才的人在哪里？" | recent | `test_ac_question_4_recent` ✅ |

**结构化证据验证**：
- `test_ac_structured_evidence`: 验证 `answer.evidence` 包含 `object_id/label/position/confidence`
- `test_answer_has_evidence_fields`: 验证证据字段来自 World Model

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_mvp_demo.py -v
# 14 passed in 0.45s
.venv\Scripts\python.exe -m pytest -q   # 全量 765 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_mvp_demo.py (14 集成测试)**：
- `TestMVPDemo` (11)：问题列表 + 处理帧 + 单次问答 + **AC 4 问题闭环** + **AC 结构化证据** + run() + print_results() + **问题1-4 分别验证**
- `TestEndToEndPipeline` (3)：完整管线 + 多帧跟踪 + 证据字段

## Metrics

| 指标 | 值 |
|---|---|
| M048 测试数 | 14 (集成测试) |
| 全量测试数 | 765 (100%) |
| M048 测试耗时 | 0.45s |
| 全量测试耗时 | 38.27s |
| Lint | All checks passed |
| Format | 3 files already formatted |
| 代码行数 | ~170 (mvp_demo) + ~160 (测试) |
| 外部依赖 | 无（纯 numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 四个问题均可完成闭环 | ✅ `test_ac_four_questions_complete_loop` 验证 4 个问题全部有答案文本 + 查询类型；`test_ac_question_1_front` / `test_ac_question_2_left` / `test_ac_question_3_distance` / `test_ac_question_4_recent` 分别验证 |
| 提供结构化证据 | ✅ `test_ac_structured_evidence` + `test_answer_has_evidence_fields` 验证 `answer.evidence` 包含 `object_id/label/position/confidence` 字段 |

## MVP 最小路径完成

```
M006✓ → M009✓ → M019✓ → M011✓ → M013✓ → M015✓ → M021✓ → M026✓
→ M029✓ → M031✓ → M036✓ → M037✓ → M038✓ → M048✓
```

**14 个 Mission 全部完成**。最小闭环已验证：
- 360 相机 → 视频帧 → 目标检测 → 深度估计 → 相机位姿 → 3D 物体
- → 世界模型 → 场景图 → 空间记忆 → LLM 上下文 → 空间问答 → 结构化证据

## Remaining Risks

- **Dummy 后端**：MVP 使用合成数据，未接入真实 360 视频/真实 YOLO 检测/真实 LLM
- **无真实 RTMP**：M003 有 RTMP 接入但 Demo 未使用
- **单线程**：无并发处理，大场景可能慢
- **无语音**：纯文本问答，M039 实时语音未实现
- **无 Web UI**：无前端界面，M049 室内 Demo 负责
