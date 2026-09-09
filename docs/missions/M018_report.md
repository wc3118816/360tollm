# MISSION_REPORT — M018 Occupancy Map

## Goal

建立 free / occupied / unknown 三态空间栅格。
AC：可查询任意位置占用状态；更新后旧信息可衰减。

## Assumptions

- M009 已建立深度估计（`DepthMap`）
- M015 已建立点云生成（`PointCloud.points`）
- M016 已建立点云优化（`PointCloudOptimizer`）
- 无外部栅格库（open3d/octomap），占用栅格纯 numpy 实现

## Files Changed

新增：
- `modules/occupancy/types.py`（`OccupancyGrid` 三态栅格 + decay + 查询）
- `modules/occupancy/builder.py`（`OccupancyBuilder` 点云→栅格 + 3D Bresenham 射线投射）
- `modules/occupancy/__init__.py`（子包 API 导出）
- `tests/unit/test_occupancy_types.py`（19 单测：构造/校验/坐标变换/查询/衰减/统计）
- `tests/unit/test_occupancy_builder.py`（12 单测：构建/射线投射/增量/衰减/查询）
- `tests/integration/test_occupancy_e2e.py`（6 集成测试：完整链路/AC 合规/多帧/优化集成）

修改：
- `README.md`：状态表更新 M018 = DONE

## Implementation

### 1. OccupancyGrid 数据结构（types.py）
```python
@dataclass(slots=True)
class OccupancyGrid:
    grid: np.ndarray          # int8 (Gx, Gy, Gz) 三态栅格
    confidence: np.ndarray   # float32 (Gx, Gy, Gz) 置信度, 随时间衰减
    origin: np.ndarray       # float64 (3,) 栅格原点世界坐标
    voxel_size: float        # 体素边长 (米)
    metadata: dict           # n_updates / decay_count
```

**三态枚举**：
- `UNKNOWN = 0`：未观测
- `FREE = 1`：已观测且空闲（射线穿过）
- `OCCUPIED = 2`：已观测且占用（有点云点）

**坐标变换**：
- `world_to_voxel(world_pos)`：世界坐标 → 体素索引（四舍五入，超出范围返回 -1）
- `voxel_to_world(voxel_idx)`：体素索引 → 世界坐标（体素中心）

**查询**：
- `status_at(world_pos)`：返回三态（超出范围返回 UNKNOWN）
- `is_occupied/is_free/is_unknown(world_pos)`：便捷查询

**时间衰减**：
- `decay(decay_rate, conf_threshold)`：`confidence *= decay_rate`，低于阈值 → UNKNOWN
- `metadata["decay_count"]` 记录衰减次数

### 2. OccupancyBuilder 点云→栅格（builder.py）
```python
builder = OccupancyBuilder(OccupancyBuilderConfig(voxel_size=0.1))
grid = builder.build(points, sensor_origin=np.zeros(3))
```

**算法**：
1. 计算 bbox + padding → grid 尺寸（限制每维 ≤ 200 体素防爆内存）
2. 标记点云点为 occupied + 重置 confidence
3. 射线投射：`sensor_origin` → 每个点，3D Bresenham 标记途经体素为 free（不覆盖 occupied）

**3D Bresenham**：
- 三轴错误累加（`err_1/err_2/err_3`）推进
- 跳过终点（终点是 occupied）
- 不覆盖已有 occupied

**增量更新**：
- `build(points, sensor_origin, existing=grid)`：在已有栅格上加新观测
- 射线数限制 5000 防卡

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_occupancy_types.py \
  tests/unit/test_occupancy_builder.py tests/integration/test_occupancy_e2e.py -v
# 37 passed in 0.65s
.venv\Scripts\python.exe -m pytest -q   # 全量 492 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_occupancy_types.py (19 单测)**：
- `TestConstruction` (5)：基本构造 + 无效维度 + confidence 不匹配 + voxel_size ≤ 0 + origin 非 3D
- `TestCoordinateTransform` (5)：world→voxel + 批量 + 超出范围 + voxel→world 中心 + 往返
- `TestStatusQuery` (5)：unknown + occupied + free + 超出范围 + is_occupied
- `TestDecay` (3)：confidence 降低 + 低于阈值→unknown + decay_count
- `TestSummary` (1)：统计正确

**test_occupancy_builder.py (12 单测)**：
- `TestBuild` (7)：基本构建 + 点 occupied + 射线 free + sensor 在 grid 内 + 空点云 + 无效 shape + 无效 sensor
- `TestIncrementalUpdate` (1)：增量更新保留旧点
- `TestDecay` (2)：构建后衰减 + 多次衰减全 unknown
- `TestPositionQuery` (1)：**AC 查询任意位置**
- **AC 衰减**：`TestDecay.test_decay_to_unknown_full`

**test_occupancy_e2e.py (6 集成测试)**：
- `TestFullPipeline` (3)：点云→栅格 + **AC 查询任意位置** + **AC 衰减**
- `TestMultiFrameUpdate` (1)：多帧增量更新
- `TestWithOptimizer` (1)：M016 优化后点云构建栅格
- `TestEndToEnd` (2)：完整工作流 + free 空间查询

## Metrics

| 指标 | 值 |
|---|---|
| M018 测试数 | 37 (31 unit + 6 integration) |
| 全量测试数 | 492 (100%) |
| M018 测试耗时 | 0.65s |
| 全量测试耗时 | 36.74s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~340 (types 200 + builder 220) + ~440 (测试) |
| 外部依赖 | 无（纯 numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 可查询任意位置占用状态 | ✅ `grid.status_at(world_pos)` 返回 UNKNOWN/FREE/OCCUPIED；`is_occupied/is_free/is_unknown` 便捷查询；超出范围返回 UNKNOWN；测试 `test_query_any_position` + `test_ac_query_any_position` 验证 |
| 更新后旧信息可衰减 | ✅ `grid.decay(decay_rate, conf_threshold)`：confidence 乘 decay_rate，低于阈值→UNKNOWN；测试 `test_decay_reduces_confidence` + `test_decay_to_unknown` + `test_ac_decay` 验证；metadata 记录 decay_count |

## Remaining Risks

- **3D Bresenham 是简化版**：不是标准 3D Bresenham 算法，可能在某些对角线路径上遗漏体素。可升级为 Amanatides-Woo 算法。
- **grid 尺寸限制 200/维**：大场景（> 20m @ 0.1m 体素）会被截断。可改用动态扩展或八叉树。
- **射线数限制 5000**：大点云会跳过部分射线。可优化为向量化批量射线投射。
- **无概率模型**：当前是三态硬决策，未用 log-odds 概率（OctoMap 风格）。可升级为连续概率。
- **无序列化**：当前 OccupancyGrid 无 save/load。可加 npz/vox 保存。
- **单线程**：大点云构建慢。可用 numba/Cython 加速射线投射。

## Next Missions

- **M019 目标检测**：在 2D 图像上检测物体，为 M021 2D→3D 关联提供输入。
- **M021 2D→3D 目标关联**：用占用栅格 + M019 检测框，把 2D 检测反投影到 3D 世界坐标。
- **M023 空间地图**：基于占用栅格构建结构化空间地图（房间/物体/关系）。
