# MISSION_REPORT — M036 World Model 序列化

## Goal

定义 LLM 可消费的稳定 JSON Schema。
必须包含：scene_id / coordinate_system / camera_pose / objects / relations / regions / occupancy / memory / confidence / timestamp。
AC：Schema 有版本号；向后兼容策略明确。

## Assumptions

- M026 已建立 SceneGraph（serialize/deserialize）
- M029 已建立 WorldModel API
- M031 已建立 SpatialMemory（serialize）

## Files Changed

新增：
- `modules/serialization/schema.py`（Schema 定义 + 版本管理 + 验证）
- `modules/serialization/serializer.py`（`WorldModelSerializer` 统一序列化器）
- `modules/serialization/__init__.py`（子包导出）
- `tests/unit/test_serialization.py`（27 单测：Schema/版本/验证/序列化）
- `tests/integration/test_serialization_e2e.py`（7 集成测试：完整链路/AC/roundtrip）

修改：
- `README.md`：状态表更新 M036 = DONE

## Implementation

### 1. Schema 定义（schema.py）
```python
SCHEMA_VERSION = "1.0.0"  # SemVer
REQUIRED_FIELDS = (10 个字段)
COORDINATE_SYSTEM = "world_xyz_meters"
```

**10 个必需字段**：
1. `scene_id`: str 场景唯一标识
2. `coordinate_system`: str 坐标系约定
3. `camera_pose`: dict 相机位姿（R + t + frame_id）
4. `objects`: list[dict] 3D 物体列表
5. `relations`: list[dict] 物体间关系
6. `regions`: list[dict] 空间区域（当前为空）
7. `occupancy`: dict 占用栅格元数据
8. `memory`: dict 空间记忆
9. `confidence`: float 全局置信度
10. `timestamp`: float 序列化时间戳

### 2. 版本管理（AC: 向后兼容策略明确）
```python
class SchemaVersion(str, Enum):
    V1_0_0 = "1.0.0"
    
    @classmethod
    def is_compatible(cls, version: str) -> bool:
        # 同 major → 兼容 (minor/patch 差异可忽略)
        # 不同 major → 不兼容
```

**向后兼容策略**：
- 新版本可添加字段，但不删除/重命名旧字段
- `deserialize` 时缺失字段填默认值（`create_empty_schema()`）
- 版本不匹配 → warning 但不报错
- 多余字段保留（forward compatible）

### 3. WorldModelSerializer（serializer.py）
```python
ser = WorldModelSerializer()
json_str = ser.serialize(world_model, scene_graph, spatial_memory, pose, "room_001")
data = ser.deserialize(json_str)  # 向后兼容
```

**序列化来源**：
- `objects` ← WorldModel.objects
- `relations` ← SceneGraph.edges
- `memory` ← SpatialMemory.objects
- `camera_pose` ← Pose (R + t + matrix)
- `confidence` ← 物体置信度均值

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_serialization.py \
  tests/integration/test_serialization_e2e.py -v
# 34 passed in 0.50s
.venv\Scripts\python.exe -m pytest -q   # 全量 696 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_serialization.py (27 单测)**：
- `TestSchema` (3)：版本格式 + 字段数 + 坐标系
- `TestSchemaVersion` (5)：当前版本 + 兼容同版本 + 兼容minor + 不兼容major + 无效
- `TestValidateSchema` (3)：合法空 + 缺失字段 + 不兼容版本
- `TestCreateEmptySchema` (2)：所有字段 + 默认值
- `TestSerializer` (14)：空序列化 + JSON字符串 + **10字段全覆盖** + **版本号** + roundtrip + **向后兼容** + 未知版本 + 不兼容版本 + pose + world_model + scene_graph + spatial_memory + 文件I/O

**test_serialization_e2e.py (7 集成测试)**：
- `TestFullPipeline` (7)：完整序列化 + **AC 版本号** + **AC 向后兼容** + **10字段全覆盖** + roundtrip + 文件持久化 + 多帧

## Metrics

| 指标 | 值 |
|---|---|
| M036 测试数 | 34 (27 unit + 7 integration) |
| 全量测试数 | 696 (100%) |
| M036 测试耗时 | 0.50s |
| 全量测试耗时 | 38.26s |
| Lint | All checks passed |
| Format | 5 files already formatted |
| 代码行数 | ~400 (schema 100 + serializer 230) + ~340 (测试) |
| 外部依赖 | 无（纯 json + time） |

## AC 对照

| AC | 实测 |
|---|---|
| Schema 有版本号 | ✅ `SCHEMA_VERSION = "1.0.0"` (SemVer)；`SchemaVersion` 枚举；序列化输出含 `schema_version` 字段；`test_schema_version_format` + `test_serialize_with_version` + `test_ac_has_version` 验证 |
| 向后兼容策略明确 | ✅ 策略：同 major 兼容、缺失字段填默认、版本不匹配 warning 不报错、多余字段保留；`SchemaVersion.is_compatible()` + `create_empty_schema()` + `deserialize` 自动填充；`test_deserialize_backward_compatible` + `test_deserialize_unknown_version` + `test_deserialize_incompatible_version` + `test_ac_backward_compatible` 验证 |

## Remaining Risks

- **regions 为空**：M025 语义地图未实现，regions 字段当前为空数组
- **occupancy 为元数据**：实际占用栅格由 M018 处理，这里只记录元数据
- **无完整反序列化恢复**：deserialize 返回 dict，未自动重建 WorldModel/SceneGraph/SpatialMemory 对象。可通过 dict 手动重建
- **无 Schema 文件**：未生成 JSON Schema 文件（如 schema.json）。可扩展
- **无压缩**：JSON 未压缩。大场景可加 gzip

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M037 LLM 上下文适配器**：把 world model JSON 转为 LLM 可读文本格式
- **M038 LLM 空间问答**：实现基础空间问答
- **M048 MVP Demo**：完成最小闭环
