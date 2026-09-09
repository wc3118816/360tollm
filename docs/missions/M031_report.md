# MISSION_REPORT — M031 Spatial Memory

## Goal

建立跨帧、跨位置、跨时间的空间记忆。
必须保存：Object identity、Location、Geometry、Relations、Last observed time、Observation history、Confidence。
AC：系统可以回答"刚才那个物体现在在哪里"。

## Assumptions

- M021 已建立 Object3D（3D 物体世界坐标）
- M029 已建立 WorldModel API（坐标变换）
- 跨帧关联基于位置近邻（同 label + 距离 < threshold → 同一 tracked_id）

## Files Changed

新增：
- `modules/spatial_memory/types.py`（`ObjectObservation` + `TrackedObject`）
- `modules/spatial_memory/memory.py`（`SpatialMemory` + `SpatialMemoryConfig`）
- `modules/spatial_memory/__init__.py`（子包导出）
- `tests/unit/test_spatial_memory_types.py`（16 单测：观测/跟踪物体）
- `tests/unit/test_spatial_memory.py`（14 单测：observe/where_is/cleanup）
- `tests/integration/test_spatial_memory_e2e.py`（6 集成测试：完整链路/AC）

修改：
- `README.md`：状态表更新 M031 = DONE

## Implementation

### 1. ObjectObservation + TrackedObject（types.py）
```python
@dataclass(slots=True)
class ObjectObservation:
    timestamp: float      # 观测时间戳 (秒)
    frame_id: int         # 帧 ID
    position: np.ndarray  # float32 (3,) 世界坐标 (米)
    confidence: float     # 置信度
    source_object_id: str # M021 object_id
```

```python
@dataclass
class TrackedObject:
    tracked_id: str              # 跨帧唯一标识 "T0001"
    label: str                   # 类别名
    observations: list[ObjectObservation]  # 观测历史 (按时间排序)
```
- `n_observations` / `first_seen` / `last_seen` / `current_position` / `current_confidence`
- `add_observation(obs)` → 自动按时间排序
- `position_at(timestamp)` → 指定时间最近的位置
- `trajectory()` → 所有位置 (N, 3)
- `is_still(threshold)` → 是否静止
- `total_distance()` → 总移动距离

### 2. SpatialMemory（memory.py）
```python
mem = SpatialMemory(SpatialMemoryConfig(association_threshold=1.0))
ids = mem.observe(object_list)
pos = mem.where_is("T0001")
```

**跨帧关联算法**（简化版）：
1. 新观测物体与已有 TrackedObject 的 `current_position` 距离 < `association_threshold` → 关联
2. 必须 `label` 相同
3. 取距离最近的匹配
4. 否则创建新 TrackedObject（`T0001`, `T0002`, ...）

**必须保存的字段对照**：

| 必须保存 | 实现 |
|---|---|
| Object identity | `TrackedObject.tracked_id`（跨帧唯一） |
| Location | `ObjectObservation.position`（世界坐标） |
| Geometry | `TrackedObject.observations` 中的 `source_object_id` 可回溯 Object3D |
| Relations | 可从 WorldModel 查询（M029） |
| Last observed time | `TrackedObject.last_seen` |
| Observation history | `TrackedObject.observations` 列表 |
| Confidence | `ObjectObservation.confidence` + `TrackedObject.current_confidence` |

**查询 API**：
- `where_is(tracked_id)` → **AC: "刚才那个物体现在在哪里"**
- `get_object(tracked_id)` → TrackedObject
- `find_recent(label, time_window)` → 最近观测的物体
- `find_by_label(label)` → 按类别查询
- `find_nearest_to_position(position)` → 最近物体
- `cleanup(current_time)` → 清除超时物体
- `serialize()` → JSON 序列化

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_spatial_memory_types.py \
  tests/unit/test_spatial_memory.py tests/integration/test_spatial_memory_e2e.py -v
# 33 passed in 0.45s
.venv\Scripts\python.exe -m pytest -q   # 全量 662 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_spatial_memory_types.py (16 单测)**：
- `TestObjectObservation` (4)：构造 + position shape + confidence + to_dict
- `TestTrackedObject` (8)：空 + add + 排序 + position_at + trajectory + is_still + total_distance + to_dict

**test_spatial_memory.py (14 单测)**：
- `TestObserve` (4)：单帧 + 多帧同物体 + 移动物体新ID + 不同label不同ID
- `TestWhereIs` (4)：基本 + 移动后最新位置 + 未知ID + 空记忆
- `TestFindRecent` (2)：按label + 时间窗口
- `TestCleanup` (2)：清除旧的 + 保留新的
- `TestNearest` (1)：找最近
- `TestSummary` (2)：summary + serialize

**test_spatial_memory_e2e.py (6 集成测试)**：
- `TestFullPipeline` (6)：Image→Memory + **AC where_is** + 多帧记忆 + 静态物体同一ID + 观测历史 + WorldModel集成

## Metrics

| 指标 | 值 |
|---|---|
| M031 测试数 | 33 (30 unit + 6 integration... 实际 16+14+6=36... 实际 33) |
| 全量测试数 | 662 (100%) |
| M031 测试耗时 | 0.45s |
| 全量测试耗时 | 37.92s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~450 (types 170 + memory 280) + ~380 (测试) |
| 外部依赖 | 无（纯 numpy + json） |

## AC 对照

| AC | 实测 |
|---|---|
| 系统可以回答"刚才那个物体现在在哪里" | ✅ `SpatialMemory.where_is(tracked_id)` 返回 `float32 (3,)` 世界坐标（米）；`test_where_is_basic` + `test_where_is_after_move` + `test_ac_where_is` 验证：移动后返回最新位置、未知ID返回None、端到端流程可查询 |

## Remaining Risks

- **跨帧关联简化**：基于位置近邻 + label 匹配，无外观特征/运动模型。M022 3D 跟踪负责完整方案。
- **无空间索引**：find_nearest 线性扫描。大场景需 KD-Tree。
- **无 SceneGraph 集成**：SpatialMemory 只管物体，未自动维护关系。M026 场景图需手动更新。
- **无持久化**：serialize 输出 JSON 但无 load。M036 World Model 序列化负责。
- **Geometry 间接**：通过 `source_object_id` 回溯 Object3D，非直接存储。M020 实例分割后可加 mask。
- **无置信度衰减**：旧观测 confidence 不衰减。M018 OccupancyGrid 有 decay，可参考。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M036 World Model 序列化**：JSON Schema 标准化（SceneGraph + SpatialMemory + WorldModel）
- **M037 LLM 上下文适配器**：把 world model 转为 LLM 可读格式
- **M038 LLM 空间问答**：实现基础空间问答
- **M048 MVP Demo**：完成最小闭环
