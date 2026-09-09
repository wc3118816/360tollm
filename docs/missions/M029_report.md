# MISSION_REPORT — M029 World Model API

## Goal

统一 World / Camera / Object 坐标系。
API：world_to_camera() / camera_to_world() / object_pose() / object_distance() / object_direction()。
AC：坐标变换双向可验证；单位统一为 SI。

## Assumptions

- M011 已建立 Pose（SE3 矩阵 world_to_camera 变换）
- M021 已建立 Object3D（3D 物体 world 坐标）
- 约定：camera: +x=forward, +y=up, +z=right；world 初始与 camera 0 重合

## Files Changed

新增：
- `modules/world_model/api.py`（`WorldModel` 统一 API）
- `modules/world_model/__init__.py`（子包导出）
- `tests/unit/test_world_model.py`（26 单测：坐标变换/物体管理/关系/SI 单位）
- `tests/integration/test_world_model_e2e.py`（6 集成测试：完整链路/AC/多帧）

修改：
- `README.md`：状态表更新 M029 = DONE

## Implementation

### WorldModel API（api.py）
```python
@dataclass
class WorldModel:
    objects: dict[str, Object3D]      # object_id → Object3D
    current_pose: Pose | None         # 当前相机位姿
    metadata: dict
```

**坐标变换**（AC: 双向可验证）：
```python
@staticmethod
def world_to_camera(p_world, pose) -> p_camera
    # p_camera = R @ p_world + t

@staticmethod
def camera_to_world(p_camera, pose) -> p_world
    # p_world = R^T @ (p_camera - t)
```

**物体查询 API**（M029 要求）：
- `object_pose(object_id)` → Object3D（位置+尺寸）
- `object_in_world(object_id)` → np.ndarray (3,) 世界坐标
- `object_in_camera(object_id, pose)` → np.ndarray (3,) 相机坐标
- `object_distance(a, b)` → float（米，SI）
- `object_direction(from, to)` → np.ndarray (3,) 单位向量
- `distance_to_camera(object_id)` → float（米）
- `direction_from_camera(object_id)` → np.ndarray (3,)
- `find_nearest(object_id, label)` → (id, distance)

**批量支持**：`world_to_camera` / `camera_to_world` 支持 (N,3) 批量变换

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_world_model.py \
  tests/integration/test_world_model_e2e.py -v
# 32 passed in 0.57s
.venv\Scripts\python.exe -m pytest -q   # 全量 629 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_world_model.py (26 单测)**：
- `TestCoordinateTransform` (8)：基本变换 + **AC roundtrip identity** + **AC roundtrip 平移** + **AC roundtrip 旋转** + **AC roundtrip 组合** + 批量 + 无效 shape
- `TestObjectManagement` (5)：add_objects + get_object + object_pose + object_in_world + object_in_camera
- `TestObjectRelations` (7)：distance + missing + direction + 同位置 + distance_to_camera 无/有 pose + direction_from_camera
- `TestFindNearest` (2)：最近 + 按类别
- `TestSIUnits` (3)：**AC 距离米** + **AC 坐标米** + **AC 方向单位向量**
- `TestSummary` (1)：summary 含 unit_length/unit_angle

**test_world_model_e2e.py (6 集成测试)**：
- `TestFullPipeline` (6)：Image→WorldModel + **AC roundtrip** + **AC SI 单位** + 物体查询 + find_nearest + 多帧更新

## Metrics

| 指标 | 值 |
|---|---|
| M029 测试数 | 32 (26 unit + 6 integration) |
| 全量测试数 | 629 (100%) |
| M029 测试耗时 | 0.57s |
| 全量测试耗时 | 45.48s |
| Lint | All checks passed |
| Format | 4 files already formatted |
| 代码行数 | ~320 (api 250) + ~330 (测试) |
| 外部依赖 | 无（纯 numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 坐标变换双向可验证 | ✅ `world_to_camera` + `camera_to_world` roundtrip：identity/平移/旋转/组合 4 种 pose 全部 `np.allclose(p_world, p_back, atol=1e-10)`；`test_roundtrip_*` + `test_ac_roundtrip` 验证 |
| 单位统一为 SI | ✅ 距离返回 float（米）；方向返回单位向量（`np.linalg.norm(direction) == 1.0`）；坐标 float32/64（米）；summary 标注 `unit_length="meter"`, `unit_angle="radian"`；`test_distance_in_meters` + `test_coordinates_in_meters` + `test_direction_unit_vector` + `test_ac_si_units` 验证 |

## Remaining Risks

- **无 Body 坐标系**：当前只有 World/Camera/Object，未实现人体骨骼 Body 坐标。需 M020 实例分割 + 人体姿态估计。
- **object_id 不跨帧**：每帧重新生成，无跨帧关联。M022 3D 跟踪负责。
- **无 SceneGraph 集成**：WorldModel 只管物体，未集成 M026 场景图。可扩展为 WorldModel.scene_graph。
- **无持久化**：当前 WorldModel 在内存。M031 Spatial Memory 负责。
- **无时间状态**：不记录物体历史位置。M030 时间状态管理负责。
- **无空间查询索引**：find_nearest 线性扫描。大场景需 KD-Tree。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M031 Spatial Memory**：持久化 WorldModel + 时间状态
- **M036 World Model 序列化**：JSON Schema 标准化
- **M037 LLM 上下文适配器**：把 world model 转为 LLM 可读格式
- **M038 LLM 空间问答**：实现基础空间问答
- **M048 MVP Demo**：完成最小闭环
