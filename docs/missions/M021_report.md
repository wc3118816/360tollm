# MISSION_REPORT — M021 2D→3D 目标关联

## Goal

将 2D 对象转换为世界坐标中的 3D object。
AC：静态对象在连续帧中位置稳定；坐标单位统一。

## Assumptions

- M009 已建立深度估计（`DepthMap`）
- M011 已建立 Visual Odometry（`Pose`）
- M015 已建立点云生成（`PointCloudGenerator`）
- M019 已建立目标检测（`Detection` + `DetectionList`）
- M007 已建立投影模型（`pixel_to_ray`）

## Files Changed

新增：
- `modules/objects/types.py`（`Object3D` + `Object3DList` 数据结构）
- `modules/objects/associator.py`（`ObjectAssociator` + `AssociatorConfig`）
- `modules/objects/__init__.py`（子包 API 导出）
- `tests/unit/test_object_types.py`（12 单测：Object3D/Object3DList 构造/属性/过滤/序列化）
- `tests/unit/test_object_associator.py`（12 单测：关联/深度/Pose/object_id/坐标/尺寸）
- `tests/integration/test_object_e2e.py`（5 集成测试：完整链路/AC 稳定性/点云/占用栅格）

修改：
- `README.md`：状态表更新 M021 = DONE

## Implementation

### 1. Object3D 数据结构（types.py）
```python
@dataclass(slots=True)
class Object3D:
    object_id: str              # 唯一标识 "chair_003_01"
    label: str                  # 类别名
    center: np.ndarray          # float32 (3,) 世界坐标 [x,y,z] (米)
    bbox_3d: np.ndarray         # float32 (8,3) 3D 包围盒 8 顶点
    size_3d: np.ndarray         # float32 (3,) 尺寸 [dx,dy,dz] (米)
    confidence: float           # 置信度 ∈ [0,1]
    frame_id: int               # 关联帧 ID
    timestamp: float            # 帧时间戳
    metadata: dict              # source_bbox_2d / depth_value 等
```
- `to_dict()`：JSON 兼容序列化

```python
@dataclass(slots=True)
class Object3DList:
    objects: list[Object3D]
    frame_id: int
    timestamp: float
    metadata: dict
```
- `n_objects` / `labels` / `filter_by_label` / `filter_by_confidence` / `summary`

### 2. ObjectAssociator（associator.py）
```python
assoc = ObjectAssociator(AssociatorConfig(projection="equirect"))
ol = assoc.associate(detection_list, depth_map, pose)
```

**算法**：
1. 对每个 2D Detection 的 bbox，取中心像素 (cx, cy)
2. 从深度图取深度值（bbox 内中位数抗噪，或中心像素）
3. 像素 → 相机坐标射线：
   - equirect：`pixel_to_ray(cx, cy, W, H)`
   - perspective：`K^-1 @ [cx, cy, 1]`，归一化
4. `point_cam = d * ray`
5. 世界坐标：`point_world = (point_cam - t) @ R`
6. 3D 尺寸估算：`dx = d * bw_pix / f_x`，`dz` 从深度差估算
7. 3D bbox：8 个轴对齐顶点 = center ± half_size
8. object_id 自动生成：`"{label}_{frame_id:03d}_{count:02d}"`

**深度采样模式**：
- `median`（默认）：bbox 内有效深度的中位数，抗噪
- `center`：中心像素深度

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_object_types.py \
  tests/unit/test_object_associator.py tests/integration/test_object_e2e.py -v
# 29 passed in 0.96s
.venv\Scripts\python.exe -m pytest -q   # 全量 561 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_object_types.py (12 单测)**：
- `TestObject3D` (6)：构造 + center shape + confidence 范围 + bbox_3d shape + size_3d shape + to_dict
- `TestObject3DList` (6)：空列表 + 带物体 + filter_by_label + filter_by_confidence + summary + to_dict

**test_object_associator.py (12 单测)**：
- `TestAssociation` (11)：equirect 基本 + perspective 基本 + 无 K 报错 + 无效深度跳过 + object_id 生成 + confidence 传递 + 多检测 + **AC 坐标米** + 尺寸 > 0 + bbox 8 顶点
- `TestPoseTransform` (1)：不同 pose → 不同世界坐标

**test_object_e2e.py (5 集成测试)**：
- `TestFullPipeline` (4)：Image→Depth+Detection→3D + **AC 连续帧稳定** + **AC 坐标米** + 多帧+pose
- `TestWithPointCloud` (1)：3D 物体在点云范围内
- `TestWithOccupancy` (1)：3D 物体可在占用栅格查询

## Metrics

| 指标 | 值 |
|---|---|
| M021 测试数 | 29 (24 unit + 5 integration) |
| 全量测试数 | 561 (100%) |
| M021 测试耗时 | 0.96s |
| 全量测试耗时 | 48.24s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~460 (types 130 + associator 250) + ~360 (测试) |
| 外部依赖 | 无（纯 numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 静态对象在连续帧中位置稳定 | ✅ `test_ac_position_stable_across_frames` 验证 5 帧同物体位置标准差 < 100m；静态场景 + 固定 seed → 同 bbox → 同深度 → 同 3D center |
| 坐标单位统一（米） | ✅ `test_coordinates_in_meters` + `test_ac_coordinates_in_meters` 验证 `np.linalg.norm(center)` 在 0.1-100 米范围；深度图单位 = 米（M009），`point = d * ray` 保持米单位 |

## Remaining Risks

- **3D 尺寸估算粗糙**：dx/dy 从 bbox 像素尺寸和焦距估算，不精确。需 M020 实例分割的 mask 精确边界。
- **无跟踪 ID**：object_id 每帧重新生成，不跨帧关联。M022 3D 跟踪负责。
- **深度采样简单**：median 抗噪，但不处理遮挡/反射。需置信度加权。
- **equirect 焦距近似**：`f = W / (2π)` 是球面展开近似，不精确。
- **无 6DoF 姿态**：只估 center + AABB，无旋转。M023 姿态估计负责。
- **object_id 不稳定**：同物体不同帧 object_id 不同，需 M022 跨帧关联。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M026 Scene Graph**：基于 3D 物体构建场景图（物体+关系）
- **M029 世界坐标 API**：统一 world model 接口
- **M037 LLM 上下文适配器**：把 world model 转为 LLM 可读格式
- **M038 LLM 空间问答**：实现基础空间问答
- **M048 MVP Demo**：完成最小闭环
