# MISSION_REPORT — M026 Semantic Scene Graph

## Goal

建立节点和边组成的场景图。
节点：Room、Door、Table、Chair、Person。
关系：inside、on、next_to、in_front_of、behind、left_of、right_of、near、far、connected_to。
AC：图结构可序列化；节点具备世界坐标；边具有置信度/来源。

## Assumptions

- M021 已建立 2D→3D 目标关联（`Object3D` + `Object3DList`）
- M025 语义地图未实现（不在 MVP 路径上），节点直接从 Object3D 转换
- 关系计算简化版（M027 空间关系计算的前置），从 3D 几何直接计算

## Files Changed

新增：
- `modules/scene_graph/types.py`（`SceneNode` + `SceneEdge` + `SceneGraph` + 序列化）
- `modules/scene_graph/builder.py`（`SceneGraphBuilder` + `GraphBuilderConfig`）
- `modules/scene_graph/__init__.py`（子包 API 导出）
- `tests/unit/test_scene_graph_types.py`（19 单测：Node/Edge/Graph/Serialize）
- `tests/unit/test_scene_graph_builder.py`（11 单测：构建/关系/增量更新）
- `tests/integration/test_scene_graph_e2e.py`（6 集成测试：完整链路/AC/roundtrip）

修改：
- `README.md`：状态表更新 M026 = DONE

## Implementation

### 1. SceneNode + SceneEdge（types.py）
```python
@dataclass(slots=True)
class SceneNode:
    node_id: str
    label: str               # chair / table / person / room / door
    position: np.ndarray     # float32 (3,) 世界坐标 [x,y,z] (米)
    size_3d: np.ndarray      # float32 (3,) 尺寸
    confidence: float        # ∈ [0,1]
    metadata: dict           # frame_id / timestamp / source
```

```python
@dataclass(slots=True)
class SceneEdge:
    source_id: str
    target_id: str
    relation: str            # inside/on/next_to/near/far/...
    confidence: float       # ∈ [0,1]
    source: str              # geometry / detection / manual
    metadata: dict           # distance / direction
```
- 关系校验：必须属于 `RELATIONS` 集合（10 种合法关系）

### 2. SceneGraph（types.py）
```python
@dataclass
class SceneGraph:
    nodes: dict[str, SceneNode]
    edges: list[SceneEdge]
    metadata: dict
```
- `add_node` / `add_edge`（自动去重）/ `get_node` / `get_neighbors` / `get_edges_from` / `get_edges_between`
- `filter_by_label` / `summary`
- **`serialize()`** → JSON 字符串（AC: 图结构可序列化）
- **`deserialize(json_str)`** → SceneGraph（反序列化恢复）

### 3. SceneGraphBuilder（builder.py）
```python
builder = SceneGraphBuilder(GraphBuilderConfig())
graph = builder.build(object_list)
```

**关系计算**（简化版，M027 前置）：
1. **near/far**：距离 < near_threshold → near；> far_threshold → far
2. **next_to**：距离 < next_to_threshold 且不在上方
3. **left_of/right_of/in_front_of/behind**：以 forward_axis（y 默认）为参考方向
4. **on**：target 在 source 上方（z 差 > on_threshold）且距离近
5. **inside**：target 在 source 的 3D bbox 内
6. 每对物体取一个最强关系（方向 > 距离 > 其他）

**增量更新**：`build(object_list, existing_graph=graph)` 支持多帧合并

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_scene_graph_types.py \
  tests/unit/test_scene_graph_builder.py tests/integration/test_scene_graph_e2e.py -v
# 36 passed in 0.41s
.venv\Scripts\python.exe -m pytest -q   # 全量 597 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_scene_graph_types.py (19 单测)**：
- `TestSceneNode` (4)：构造 + position shape + confidence 范围 + to_dict
- `TestSceneEdge` (4)：构造 + relation 合法性 + confidence 范围 + to_dict
- `TestSceneGraph` (7)：空图 + add_node + add_edge + **边去重** + get_neighbors + get_edges_from + filter_by_label + summary
- `TestSerialize` (3)：**AC 序列化往返** + **AC 合法 JSON** + to_dict

**test_scene_graph_builder.py (11 单测)**：
- `TestBuild` (5)：基本构建 + 多物体 + 空列表 + **AC 节点世界坐标** + **AC 边置信度/来源**
- `TestRelations` (5)：near + far + direction + on + inside
- `TestIncremental` (1)：增量更新

**test_scene_graph_e2e.py (6 集成测试)**：
- `TestFullPipeline` (6)：Image→SceneGraph + **AC 序列化** + **AC 节点坐标** + **AC 边置信度/来源** + roundtrip + 多帧增量

## Metrics

| 指标 | 值 |
|---|---|
| M026 测试数 | 36 (30 unit + 6 integration) |
| 全量测试数 | 597 (100%) |
| M026 测试耗时 | 0.41s |
| 全量测试耗时 | 43.86s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~520 (types 240 + builder 180) + ~380 (测试) |
| 外部依赖 | 无（纯 numpy + json） |

## AC 对照

| AC | 实测 |
|---|---|
| 图结构可序列化 | ✅ `SceneGraph.serialize()` → JSON 字符串；`deserialize(json_str)` → SceneGraph；`test_serialize_deserialize_roundtrip` + `test_ac_serializable` 验证 roundtrip 保持节点/边/元数据；`test_serialize_valid_json` 验证输出是合法 JSON |
| 节点具备世界坐标 | ✅ `SceneNode.position` float32 (3,) 米单位；`test_ac_node_has_world_coords` + `test_ac_node_world_coords` 验证 `np.allclose(node.position, obj.center)` 且距离在 0.1-100m |
| 边具有置信度/来源 | ✅ `SceneEdge.confidence` ∈ [0,1] + `SceneEdge.source` = "geometry"/"detection"/"manual"；`test_ac_edge_has_confidence_source` + `test_ac_edge_confidence_source` 验证所有边有置信度和来源 |

## Remaining Risks

- **关系计算简化**：每对物体只取一个最强关系（方向 > 距离），实际场景中两个物体可能同时是 near + left_of。M027 会完善多关系。
- **无 connected_to**：需外部标注（房间/门拓扑关系），当前不计算。
- **无 Room/Door 节点**：M025 语义地图未实现，当前只有 Object3D 转换的物体节点。
- **object_id 不跨帧**：每帧重新生成 ID，无跨帧关联。M022 3D 跟踪负责。
- **无图查询优化**：当前线性遍历边。大图需索引加速。
- **关系来源单一**：当前所有边 source="geometry"。M027+ 会加 "detection"（来自检测框重叠）和 "manual"（用户标注）。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M029 世界坐标与对象坐标 API**：统一 world model 接口
- **M031 Spatial Memory**：持久化场景图 + 时间状态
- **M036 World Model 序列化**：JSON Schema 标准化
- **M037 LLM 上下文适配器**：把 world model 转为 LLM 可读格式
- **M038 LLM 空间问答**：实现基础空间问答
- **M048 MVP Demo**：完成最小闭环
