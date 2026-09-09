# MISSION_REPORT — M015 点云生成

## Goal

生成3D点云，用于空间理解与可视化。AC：点云在 world 坐标系对齐；无 NaN。

## Assumptions

- M009 已建立深度估计（`DepthMap`，float32 (H,W) 米）
- M011 已建立 Visual Odometry（`Pose`，4x4 SE3 world_to_camera）
- M007 已建立投影模型（`pixel_to_ray` 用于 equirect 反投影）
- M006 标定可选（perspective 模式需内参 K）
- 无外部点云库（open3d/pypcd），PLY 序列化自行实现

## Files Changed

新增：
- `modules/pointcloud/types.py`（`PointCloud` 数据结构 + PLY 保存/加载）
- `modules/pointcloud/generator.py`（`PointCloudGenerator` + `voxel_downsample`）
- `modules/pointcloud/__init__.py`（子包 API 导出）
- `tests/unit/test_pointcloud_types.py`（16 单测：构造/校验/NaN/merge/summary/PLY）
- `tests/unit/test_pointcloud_generator.py`（19 单测：equirect/perspective/过滤/颜色/多帧/体素）
- `tests/integration/test_pointcloud_e2e.py`（8 集成测试：完整链路/AC 合规/PLY 往返）

修改：
- `README.md`：状态表更新 M015 = DONE

## Implementation

### 1. PointCloud 数据结构（types.py）
```python
@dataclass(slots=True)
class PointCloud:
    points: np.ndarray   # float32 (N, 3) 世界坐标
    colors: np.ndarray  # uint8 (N, 3) RGB, 可选
    normals: np.ndarray # float32 (N, 3), 可选
    metadata: dict      # source_frame_id / depth_backend / projection 等
```
- `has_nan`：检查 points 是否含 NaN/Inf（AC: 点云无 NaN）
- `filter_valid()`：移除 NaN/Inf 点（in-place）
- `merge()`：合并另一个点云（in-place）
- `summary()`：统计摘要（n_points / has_nan / bbox_min/max）
- `save_ply()` / `load_ply()`：PLY 二进制 + ASCII 序列化（无外部依赖）

### 2. PointCloudGenerator（generator.py）
```python
gen = PointCloudGenerator(PointCloudGeneratorConfig(projection="equirect"))
pc = gen.generate(image, depth_map, pose, camera_matrix=None)  # 单帧
pc = gen.generate_multi(frames, camera_matrix=None)            # 多帧融合
```
**两种投影模式**：
- `equirect`（默认）：用 M007 的 `pixel_to_ray(u, v, W, H)` 把全景像素反投影成 3D 射线
- `perspective`：用内参 K 反投影 `ray = K^-1 @ [u, v, 1]`

**世界坐标变换**：
- `matrix` 是 world_to_camera，其逆是 camera_to_world
- `point_world = R^T @ (point_cam - t)` 等价于 `(point_cam - t) @ R`（因 R 正交）
- 多帧融合：对每帧生成局部点云 → 变换到 world → vstack 合并

**深度过滤**：
- `depth > min_depth`（默认 0.1m）
- `depth < max_depth`（默认 100m）
- `np.isfinite(depth)`（移除 NaN/Inf）
- `filter_nan`：最终点云再过滤一次（防止 pose 变换引入 NaN）

**降采样**：
- `step` 参数：每隔 step 个像素取一个点（step=2 → 1/4 点数）
- `voxel_downsample(points, voxel_size)`：体素滤波，每体素取均值

### 3. PLY 序列化（types.py）
- 二进制：numpy structured array + `tobytes()`（紧凑）
- ASCII：逐行写 `x y z [r g b]`（可读）
- 加载：解析 header + `np.frombuffer`（binary）或 `np.array`（ASCII）
- 无需 open3d/pypcd，纯 numpy + struct

## Tests

```bash
uv run pytest tests/unit/test_pointcloud_types.py tests/unit/test_pointcloud_generator.py \
  tests/integration/test_pointcloud_e2e.py -v
# 43 passed in 0.60s
uv run pytest -q   # 全量 379 passed
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖

**test_pointcloud_types.py (16 单测)**：
- `TestPointCloudConstruction` (3)：基本构造 + 带颜色 + 空
- `TestPointCloudValidation` (3)：形状错误 + 颜色长度不一致 + 法向量长度不一致
- `TestPointCloudNaN` (3)：has_nan true/false + filter_valid
- `TestPointCloudMerge` (2)：合并两个 + 带颜色合并
- `TestPointCloudSummary` (1)：summary 字段
- `TestPointCloudPLY` (4)：binary 保存/加载 + ASCII + 无颜色 + 文件创建

**test_pointcloud_generator.py (19 单测)**：
- `TestEquirectGeneration` (5)：返回 PointCloud + 点数匹配 + 无 NaN + identity pose 世界坐标 + 平移 pose
- `TestPerspectiveGeneration` (3)：带 K 生成 + 无 K 报错 + 中心像素反投影
- `TestDepthFiltering` (3)：零深度过滤 + max_depth 过滤 + step 降采样
- `TestColorExtraction` (2)：颜色提取 + 无图无颜色
- `TestMultiFrameFusion` (2)：多帧合并 + 空输入
- `TestVoxelDownsample` (4)：点数减少 + 范围保留 + 带颜色 + 空输入

**test_pointcloud_e2e.py (8 集成测试)**：
- `TestFullPipeline` (3)：Frame→PointCloud 完整链路 + **AC 无 NaN** + **AC world 对齐**
- `TestMultiFrameFusion` (2)：多帧点数增加 + metadata
- `TestPLERoundTrip` (2)：PLY 保存/加载往返 + 文件存在
- `TestEndToEndWithVO` (1)：完整工作流（Image→Depth+VO→PointCloud→PLY）

## Metrics

| 指标 | 值 |
|---|---|
| M015 测试数 | 43 (35 unit + 8 integration) |
| 全量测试数 | 379 (100%) |
| M015 测试耗时 | 0.60s |
| 全量测试耗时 | 41.63s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~280 (types 200 + generator 230) + ~520 (测试) |
| 外部依赖 | 无（纯 numpy + struct） |

## AC 对照

| AC | 实测 |
|---|---|
| 点云在 world 坐标系对齐 | ✅ `generate()` 把 camera 坐标用 `Pose.R/t` 变换到 world；`generate_multi()` 多帧各自变换后合并；测试 `test_ac_world_coords_aligned` 验证两帧不同 pose → 不同 world 位置（centroid 偏移 > 0.5） |
| 无 NaN | ✅ `PointCloud.has_nan` + `filter_nan` 配置项 + `generate()` 中 `np.isfinite` 过滤；测试 `test_ac_no_nan` 验证；深度过滤移除无效值 |

## Remaining Risks

- **深度质量决定点云质量**：Dummy 深度是合成的，点云形状不反映真实几何。真实场景需 MiDaS / Metric3D。
- **单目尺度模糊**：DummyVO 的合成轨迹是真实尺度，但 FeatureVO（ORB）的尺度是启发式。M012 VIO 会修正。
- **全景图深度**：MiDaS 是透视图训练的，对 equirect 有畸变。全景专用深度模型留后续。
- **无法向量计算**：当前 normals 为空。M016 法向量估计可用 PCA 局部拟合或深度图梯度。
- **无点云配准**：多帧融合假设 pose 已对齐。真实场景需 ICP 或回环检测（M014）修正漂移。
- **体素降采样是 O(N) 实现**：用 `np.unique` + 循环均值，对大点云（百万点）可能慢。可优化为 `np.add.at` + `np.bincount`。

## Next Missions

- **M016 点云优化**：法向量估计 + 噪声滤波 + 表面重建（Poisson / Ball Pivoting）。
- **M021 2D→3D 目标关联**：用 M015 点云 + M019 检测框，把 2D 检测反投影到 3D 世界坐标。
- **M023 空间地图**：基于 M015 点云构建结构化空间地图（房间/物体/关系）。
