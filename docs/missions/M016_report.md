# MISSION_REPORT — M016 点云优化

## Goal

对 M015 生成的点云进行优化：估计法向量、移除离群点、整合为统一优化管线。
AC：点云无 NaN 值；离群点被移除。

## Assumptions

- M015 已建立 `PointCloud` 数据结构（points + colors + normals + PLY 序列化）
- M015 已实现 `voxel_downsample` 体素降采样
- 无外部点云库（open3d/pypcd），法向量与滤波纯 numpy 实现
- 点云规模 < 50k 点（brute-force kNN O(N²) 适用范围）

## Files Changed

新增：
- `modules/pointcloud/normals.py`（法向量估计：PCA 局部拟合 + k 近邻 + 视点定向）
- `modules/pointcloud/filter.py`（统计滤波 + 半径滤波 + PointCloud 适配）
- `modules/pointcloud/optimizer.py`（`PointCloudOptimizer` 链式管线 + `OptimizerConfig`）
- `tests/unit/test_pointcloud_normals.py`（10 单测：平面/球面法向量 + 属性 + PointCloud 适配）
- `tests/unit/test_pointcloud_filter.py`（14 单测：统计/半径滤波 + PointCloud 适配 + metadata）
- `tests/integration/test_pointcloud_optimize_e2e.py`（8 集成测试：完整链路 + AC 合规 + PLY 往返）

修改：
- `modules/pointcloud/__init__.py`：导出 M016 公开 API（优化器 + 法向量 + 滤波）
- `README.md`：状态表更新 M016 = DONE

## Implementation

### 1. 法向量估计（normals.py）
```python
normals = estimate_normals(points, k=10, viewpoint=None)  # (N, 3) float32
pc = estimate_normals_pc(pc, k=10, viewpoint=None)        # PointCloud 适配
```
**算法**：
- 对每个点找 k 近邻（brute-force，纯 numpy 距离矩阵 + `argpartition`）
- 计算局部协方差矩阵（3×3）：`cov = centered.T @ centered / k`
- `np.linalg.eigh` 特征分解（升序），最小特征值对应的特征向量 = 法向量
- 归一化为单位向量
- 定向：法向量朝向视点（viewpoint）或局部质心，用点积符号判定

### 2. 离群点滤波（filter.py）
```python
mask = statistical_outlier_removal(points, k=10, n_sigma=2.0)  # 统计滤波
mask = radius_outlier_removal(points, radius=0.1, min_neighbors=5)  # 半径滤波
pc = filter_statistical_pc(pc, k=10, n_sigma=2.0)  # PointCloud 适配
pc = filter_radius_pc(pc, radius=0.1, min_neighbors=5)
```
**统计滤波**：
- 每个点到 k 近邻的平均距离
- 全局 `mean + n_sigma * std` 作为阈值
- 超过阈值的点视为离群

**半径滤波**：
- 每个点在半径 r 内的邻点数（含自身）
- 少于 `min_neighbors` 的点移除

**PointCloud 适配**：
- `inplace` 参数：True 直接修改，False 返回副本
- 同步过滤 points / colors / normals
- metadata 记录 `n_before` / `n_after` / 参数

### 3. PointCloudOptimizer（optimizer.py）
```python
opt = PointCloudOptimizer(OptimizerConfig(voxel_size=0.05, enable_normals=True))
opt_pc = opt.optimize(pc)
```
**链式流程**：
1. 体素降采样（复用 M015 `voxel_downsample`，减少点数）
2. 统计滤波（移除离群点）
3. 半径滤波（移除孤立点）
4. 法向量估计（PCA 局部拟合）

每步可选，通过 `OptimizerConfig` 控制。metadata 记录 `optimized` / `n_before_optimize` / `n_after_optimize`。

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_pointcloud_normals.py \
  tests/unit/test_pointcloud_filter.py \
  tests/integration/test_pointcloud_optimize_e2e.py -v
# 32 passed in 0.59s
.venv\Scripts\python.exe -m pytest -q   # 全量 411 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_pointcloud_normals.py (10 单测)**：
- `TestEstimateNormalsPlane` (2)：XY 平面法向量朝 Z + XZ 平面法向量朝 Y
- `TestEstimateNormalsSphere` (1)：球面法向量径向
- `TestNormalProperties` (4)：单位长度 + 空点云 + 单点 + 小 k
- `TestEstimateNormalsPC` (3)：添加法向量到 PointCloud + metadata + 非 inplace

**test_pointcloud_filter.py (14 单测)**：
- `TestStatisticalOutlierRemoval` (4)：移除离群 + 无离群全保留 + 空点云 + 严格阈值移除更多
- `TestRadiusOutlierRemoval` (4)：移除孤立点 + 空点云 + 大半径全保留 + 大阈值全移除
- `TestFilterPointCloud` (6)：统计/半径 PointCloud 适配 + 颜色保留 + metadata + 非 inplace + 空点云

**test_pointcloud_optimize_e2e.py (8 集成测试)**：
- `TestFullOptimizePipeline` (3)：生成→优化 + **AC 法向量已估计** + **AC 离群点被移除**
- `TestOptimizerSteps` (3)：体素降采样减少点数 + 滤波减少点数 + metadata 更新
- `TestPLYWithNormals` (1)：带法向量点云 PLY 保存/加载
- `TestEndToEndWorkflow` (1)：完整工作流（Image→Depth+VO→PointCloud→Optimize→PLY）

## Metrics

| 指标 | 值 |
|---|---|
| M016 测试数 | 32 (24 unit + 8 integration) |
| 全量测试数 | 411 (100%) |
| M016 测试耗时 | 0.59s |
| 全量测试耗时 | 41.54s |
| Lint | All checks passed |
| Format | 9 files already formatted |
| 代码行数 | ~437 (normals 144 + filter 182 + optimizer 111) + ~471 (测试) |
| 外部依赖 | 无（纯 numpy） |

## AC 对照

| AC | 实测 |
|---|---|
| 点云无 NaN 值 | ✅ 优化器各步在有限点集上操作；法向量由 `np.linalg.eigh` 对有限协方差矩阵计算；滤波基于距离矩阵不引入 NaN；M015 `filter_valid()` 已移除输入 NaN |
| 离群点被移除 | ✅ `statistical_outlier_removal` 按 `mean + n_sigma * std` 阈值移除；`radius_outlier_removal` 按半径内邻点数移除；集成测试 `test_ac_outliers_removed` 验证添加离群点后优化点数减少 |

## Remaining Risks

- **brute-force kNN 是 O(N²)**：法向量估计和滤波均用全距离矩阵，对大点云（> 50k 点）会爆内存。可升级为 `scipy.spatial.cKDTree`。
- **法向量定向依赖视点/质心**：复杂几何（凹面、薄壁）可能定向不一致。可引入 MST（最小生成树）传播定向。
- **滤波参数需调优**：`n_sigma` / `radius` / `min_neighbors` 依赖场景，默认值可能不适合所有数据。
- **体素降采样先于滤波**：当前顺序固定（降采样→统计→半径→法向量），不可重排。部分场景可能需先滤波再降采样。
- **未做点云配准**：M015 多帧融合假设 pose 已对齐，真实场景需 ICP 或回环检测修正漂移（M014）。
- **PLY 不保存法向量**：当前 PLY 格式只存 points + colors，法向量需单独保存或扩展 PLY 格式。

## Next Missions

- **M017 Mesh 重建**：基于点云 + 法向量生成三角网格（Poisson / Ball Pivoting），导出 glTF/PLY/OBJ。
- **M014 回环检测**：修正 VO 漂移，提升多帧融合精度。
- **M021 2D→3D 目标关联**：用优化后点云 + M019 检测框，把 2D 检测反投影到 3D 世界坐标。
