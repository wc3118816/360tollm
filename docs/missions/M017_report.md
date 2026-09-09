# MISSION_REPORT — M017 Mesh 重建

## Goal

从深度图生成三角网格，用于可视化与碰撞检测。AC：mesh 可导出 glTF/PLY/OBJ；破洞率和三角面数可统计。

## Assumptions

- M009 已建立深度估计（`DepthMap`，float32 (H,W) 米）
- M011 已建立 Visual Odometry（`Pose`，4x4 SE3）
- M007 已建立投影模型（`pixel_to_ray` 用于 equirect 反投影）
- 无外部网格库（open3d/trimesh），PLY/OBJ/glTF 序列化自行实现

## Files Changed

新增：
- `modules/mesh/types.py`（`TriangleMesh` 数据结构 + PLY/OBJ/glTF 导出/加载）
- `modules/mesh/generator.py`（`MeshGenerator` 深度图→网格 + `MeshGeneratorConfig`）
- `modules/mesh/__init__.py`（子包 API 导出）
- `tests/unit/test_mesh_types.py`（19 单测：构造/校验/属性/法向量/PLY/OBJ/glTF）
- `tests/unit/test_mesh_generator.py`（13 单测：equirect/perspective/破洞/颜色/降采样/pose）
- `tests/integration/test_mesh_e2e.py`（12 集成测试：完整链路/AC 合规/多格式导出/PLY 往返）

修改：
- `README.md`：状态表更新 M017 = DONE

## Implementation

### 1. TriangleMesh 数据结构（types.py）
```python
@dataclass(slots=True)
class TriangleMesh:
    vertices: np.ndarray        # float32 (V, 3) 顶点坐标
    faces: np.ndarray           # int32 (F, 3) 三角形顶点索引
    vertex_normals: np.ndarray  # float32 (V, 3) 顶点法向量, 可选
    vertex_colors: np.ndarray   # uint8 (V, 3) 顶点颜色, 可选
    metadata: dict              # projection / hole_ratio / n_holes 等
```
- `compute_face_normals()`：叉积计算面法向量
- `summary()`：n_vertices / n_faces / has_nan / hole_ratio
- `has_nan()`：检查顶点是否含 NaN/Inf

### 2. MeshGenerator 深度图网格三角化（generator.py）
```python
gen = MeshGenerator(MeshGeneratorConfig(projection="equirect"))
mesh = gen.generate(depth_map, pose, image=None, camera_matrix=None)
```
**算法：有序网格三角化（Organized Grid Triangulation）**
- 深度图 (H, W) 天然有序，每个像素对应一个 3D 顶点
- 每个 2x2 像素块生成 2 个三角形：`[v0, v1, v2]` 和 `[v1, v3, v2]`
- 无效深度（≤ min_depth 或 > max_depth 或 NaN）的顶点跳过
- 深度跳变（相邻顶点距离 > max_edge_length）的三角形跳过 → 破洞

**两种投影模式**：
- `equirect`（默认）：用 M007 的 `pixel_to_ray(u, v, W, H)` 反投影
- `perspective`：用内参 K 反投影 `ray = K^-1 @ [u, v, 1]`

**世界坐标变换**：`point_world = (point_cam - t) @ R`（R 正交）

### 3. 导出格式（types.py）

**PLY**（vertices + faces, binary + ASCII）：
- binary：numpy structured array + `struct.pack` 紧凑
- ASCII：逐行写 `x y z [nx ny nz] [r g b]` + `3 i0 i1 i2`
- 加载：解析 header + `np.frombuffer`（binary）或 `np.array`（ASCII）

**OBJ**：
- `v x y z` 顶点（1-indexed）
- `vn nx ny nz` 法向量
- `f i0//n0 i1//n1 i2//n2` 面（带法向量）/ `f i0 i1 i2`（无）

**glTF 2.0**（JSON + .bin 文件）：
- buffer：vertices（float32）+ indices（uint32）
- bufferView 0：ARRAY_BUFFER（vertices）
- bufferView 1：ELEMENT_ARRAY_BUFFER（indices）
- accessor：POSITION（VEC3 FLOAT）+ indices（SCALAR UNSIGNED_INT）
- min/max 自动计算

### 4. 破洞率 + 三角面数统计
```python
hole_ratio = n_holes / total_possible_faces
# total_possible = (H-1) * (W-1) * 2
# n_holes = total_possible - n_faces
```
- `mesh.summary()` 返回 `n_faces` + `hole_ratio`
- `mesh.metadata["hole_ratio"]` 和 `mesh.metadata["n_holes"]`

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_mesh_types.py \
  tests/unit/test_mesh_generator.py tests/integration/test_mesh_e2e.py -v
# 44 passed in 0.56s
.venv\Scripts\python.exe -m pytest -q   # 全量 455 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_mesh_types.py (19 单测)**：
- `TestMeshConstruction` (6)：基本构造 + 带颜色法向量 + 空 + 无效 shape + 索引越界
- `TestMeshProperties` (4)：顶点/面数 + has_nan true/false + summary
- `TestFaceNormals` (2)：叉积计算 + 空面法向量
- `TestPLYExport` (3)：binary 往返 + ASCII 往返 + 带颜色
- `TestOBJExport` (2)：基本导出 + 带法向量
- `TestGLTFExport` (2)：基本导出 + min/max 验证

**test_mesh_generator.py (13 单测)**：
- `TestMeshGeneration` (4)：equirect 基本 + perspective 基本 + 面数公式 + 无 K 报错
- `TestHoleRatio` (3)：无破洞 + 无效深度破洞 + 深度跳变破洞
- `TestMeshProperties` (6)：顶点颜色 + 无图无颜色 + 降采样 + metadata + 无 NaN + pose 变换 + 索引有效

**test_mesh_e2e.py (12 集成测试)**：
- `TestFullPipeline` (2)：Image→Depth→Mesh + **AC 无 NaN**
- `TestExportFormats` (4)：**AC glTF/PLY/OBJ 导出** + 三格式一致
- `TestStatistics` (2)：**AC 三角面数统计** + **AC 破洞率统计**
- `TestPLYRoundTrip` (1)：PLY 保存/加载往返
- `TestEndToEndWorkflow` (2)：完整工作流（多帧+多格式）+ 透视投影

## Metrics

| 指标 | 值 |
|---|---|
| M017 测试数 | 44 (32 unit + 12 integration) |
| 全量测试数 | 455 (100%) |
| M017 测试耗时 | 0.56s |
| 全量测试耗时 | 41.96s |
| Lint | All checks passed |
| Format | 6 files already formatted |
| 代码行数 | ~460 (types 310 + generator 230) + ~500 (测试) |
| 外部依赖 | 无（纯 numpy + struct + json） |

## AC 对照

| AC | 实测 |
|---|---|
| mesh 可导出 glTF/PLY/OBJ | ✅ `save_gltf()` / `save_ply()` / `save_obj()` 三种格式；集成测试 `test_export_gltf/ply/obj` 验证文件生成；`test_all_formats_consistent` 验证三格式顶点数/面数一致 |
| 破洞率可统计 | ✅ `mesh.metadata["hole_ratio"]` + `mesh.summary()["hole_ratio"]`；测试 `test_hole_ratio_stat` 验证 ∈ [0, 1]；`test_invalid_depth_creates_holes` + `test_depth_jump_creates_holes` 验证破洞检测 |
| 三角面数可统计 | ✅ `mesh.n_faces` + `mesh.summary()["n_faces"]`；测试 `test_face_count_stat` 验证 > 0；`test_face_count_formula` 验证 = (H-1)*(W-1)*2 |

## Remaining Risks

- **有序网格三角化依赖深度图结构**：仅适合从深度图直接生成 mesh。散乱点云需 Ball Pivoting / Poisson 重建（留后续）。
- **brute-force 射线计算**：equirect 模式用列表推导式调用 `pixel_to_ray`，大图慢。可向量化批量计算。
- **无顶点法向量**：当前 mesh 无顶点法向量（仅面法向量）。可用 M016 的 PCA 法向量估计补充。
- **glTF 不含颜色/法向量**：当前 glTF 只导出 POSITION + indices。可扩展 COLOR_0 / NORMAL 属性。
- **PLY face 用逐个 struct.pack**：大 mesh 慢。可优化为 numpy structured array 批量写入。
- **无 LOD/简化**：mesh 面数 = (H-1)*(W-1)*2，大图会很大。可加 quadric edge collapse 简化。

## Next Missions

- **M018 Occupancy Map**：基于 mesh/点云建立 free/occupied/unknown 三态空间栅格。
- **M021 2D→3D 目标关联**：用 mesh + M019 检测框，把 2D 检测反投影到 3D 世界坐标。
- **M023 空间地图**：基于 mesh 构建结构化空间地图（房间/物体/关系）。
