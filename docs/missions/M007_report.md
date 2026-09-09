# MISSION_REPORT — M007 全景投影模型

## Goal

提供 equirectangular ↔ spherical ↔ camera-ray 的统一数学模型。AC：已知测试点的角度误差通过阈值；数学单元测试覆盖。

## Assumptions

- M006 已完成（相机标定，提供透视相机内参 K）。
- DJI Osmo 360 输出 equirectangular 全景图（典型 2:1 宽高比，例如 7680×3840 或 2048×1024）。
- 模块为纯数学，无外部数据依赖（仅 numpy）。
- 右手坐标系约定：+x=forward, +y=up, +z=right。

## Files Changed

新增：
- `modules/projection/__init__.py`（子包 API 导出）
- `modules/projection/spherical.py`（equirect pixel ↔ (yaw,pitch) ↔ 3D ray）
- `modules/projection/rotation.py`（yaw/pitch/roll 旋转矩阵 + R_ypr 组合）
- `modules/projection/fov.py`（perspective hfov/vfov + equirect 整图/子区域 FOV + 立体角）
- `tests/unit/test_projection_spherical.py`（45 单测：已知点/往返/直连/极点）
- `tests/unit/test_projection_rotation.py`（33 单测：基本旋转/正交性/R_ypr/与 spherical 一致）
- `tests/unit/test_projection_fov.py`（16 单测：透视 FOV/反函数/equirect 整图/立体角）
- `tests/integration/test_projection_e2e.py`（15 集成测试：像素→射线→旋转→像素/AC 角度误差/全链路往返）

修改：
- `README.md`：状态表更新 M007 = DONE

## Implementation

### 坐标系与约定（spherical.py docstring）

```
右手坐标系:
    +x = forward (yaw=0, pitch=0 时的方向)
    +y = up
    +z = right (面向 +x 时的右手方向)

球面坐标:
    yaw   ∈ [-π, π]     方位角 (绕 +y), 0 = +x (前), +π/2 = -z (左), -π/2 = +z (右)
    pitch ∈ [-π/2, π/2]  俯仰角 (绕 +z), 0 = 地平, +π/2 = +y (上), -π/2 = -y (下)

Equirect 图像 (W x H):
    u = W/2 → yaw = 0;  u = 0 → yaw = +π;  u = W → yaw = -π
    v = H/2 → pitch = 0; v = 0 → pitch = +π/2; v = H → pitch = -π/2

3D 射线:
    r = (cos(pitch)*cos(yaw), sin(pitch), -cos(pitch)*sin(yaw))
```

### 1. 球面坐标变换（spherical.py）

```python
# Equirect pixel ↔ (yaw, pitch)
yaw, pitch = pixel_to_spherical(u, v, W, H)
u, v = spherical_to_pixel(yaw, pitch, W, H)

# (yaw, pitch) ↔ 3D unit ray
ray = spherical_to_ray(yaw, pitch)            # (3,) float64
yaw, pitch = ray_to_spherical(ray)            # 接受未归一化输入

# 直连便捷函数
ray = pixel_to_ray(u, v, W, H)
u, v = ray_to_pixel(ray, W, H)
```

关键点：
- `spherical_to_pixel` 对 u 做 `% W` wrap（处理后向 yaw=±π 的双重映射）
- `ray_to_spherical` 对 y 做 `asin` 截断（防浮点误差让 |y| > 1）
- 零向量返回 (0, 0)

### 2. 旋转矩阵（rotation.py）

```python
R = rotation_yaw(yaw)        # 绕 +y
R = rotation_pitch(pitch)    # 绕 +z
R = rotation_roll(roll)      # 绕 +x
R = rotation_ypr(yaw, pitch, roll=0)  # = R_yaw @ R_pitch @ R_roll
```

关键一致性（AC 核心）：
```python
# rotation_ypr 与 spherical_to_ray 必须给出同一前向方向
R = rotation_ypr(yaw, pitch, 0)
forward = R @ [1, 0, 0]
# forward == spherical_to_ray(yaw, pitch)
```
测试 `TestConsistencyWithSpherical` 对 10 个已知点 + 50 个随机点验证此一致性，全部通过。

### 3. FOV 计算（fov.py）

```python
# 透视相机
hfov = perspective_hfov(fx, width)         # 2*atan(W/(2*fx))
vfov = perspective_vfov(fy, height)
fx = focal_from_hfov(hfov, width)          # 反函数

# Equirect
full_hfov = equirect_full_hfov()           # 2π
full_vfov = equirect_full_vfov()           # π
dyaw, dpitch = equirect_region_angular_extent(u_min, u_max, v_min, v_max, W, H)
omega = equirect_region_solid_angle(u_min, u_max, v_min, v_max, W, H)  # 球面积分
```

立体角公式：`Ω = (sin(pitch_max) - sin(pitch_min)) * (yaw_max - yaw_min)`，整图 = 4π（整球）。

## Tests

```bash
uv run pytest tests/unit/test_projection_spherical.py \
  tests/unit/test_projection_rotation.py \
  tests/unit/test_projection_fov.py \
  tests/integration/test_projection_e2e.py -v
# 116 passed in 0.47s

uv run pytest -q   # 全量 239 passed
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖

**test_projection_spherical.py (45 单测)**：
- `TestPixelToSphericalKnownPoints` (9)：AC 已知像素点 → 已知 (yaw, pitch)，角度误差 < 1e-6°
- `TestSphericalToPixelRoundTrip` (9)：pixel → spherical → pixel 往返一致（含边界 wrap）
- `TestSphericalToRayKnown` (6)：前/后/左/右/上/下 6 个已知方向
- `TestRayToSpherical` (5)：前/左/上/零向量/未归一化输入
- `TestSphericalRayRoundTrip` (2)：50 个随机点往返 + 单位长度不变
- `TestPixelRayDirect` (7)：pixel → ray 直连已知点
- `TestRayToPixel` (5)：ray → pixel 已知映射

**test_projection_rotation.py (33 单测)**：
- `TestRotationYaw/Pitch/Roll`：单位矩阵/90° 变换/不变轴
- `TestOrthogonality`：正交性 (R @ R^T = I) + 行列式 = 1 + 逆 = 转置
- `TestRotationYPR`：组合矩阵性质 + 矩阵乘法顺序
- `TestConsistencyWithSpherical` (11)：**AC 关键** — R_ypr @ (1,0,0) == spherical_to_ray

**test_projection_fov.py (16 单测)**：
- `TestPerspectiveFOV`：90° 已知点 + 反函数 + 大 fx 窄 FOV
- `TestEquirectFullFOV`：整图 2π × π
- `TestEquirectRegionExtent`：半图/四分之一/整图角范围
- `TestEquirectSolidAngle`：整图 4π + 赤道带 + 极点小于赤道 + 零高度

**test_projection_e2e.py (15 集成测试)**：
- `TestRotatePixelToPixel` (4)：像素 → 射线 → 旋转 → 像素，验证已知映射（中心→左/右/上，左→中心）
- `TestYPRConsistency` (5)：跨模块 R_ypr 与 spherical_to_ray 一致 + roll 不改变前向
- `TestAngularErrorThreshold`：**AC** — 100 个随机点往返角度误差 < 1e-6°
- `TestFullPipelineRoundTrip` (5)：pixel → spherical → ray → spherical → pixel 全链路

## Metrics

| 指标 | 值 |
|---|---|
| M007 测试数 | 116 (94 unit + 15 integration + 7 known-point params) |
| 全量测试数 | 239 (100%) |
| M007 测试耗时 | 0.47s |
| 全量测试耗时 | 40.16s |
| Lint | All checks passed |
| Format | 8 files already formatted |
| 代码行数 | ~250 (spherical 130 + rotation 75 + fov 90) + ~650 (测试) |
| 外部依赖 | 仅 numpy (无 opencv / 无新增依赖) |

## AC 对照

| AC | 实测 |
|---|---|
| 已知测试点的角度误差通过阈值 | ✅ `TestPixelToSphericalKnownPoints` 9 个已知点角度误差 < 1e-6°；`TestAngularErrorThreshold` 100 随机点往返误差 < 1e-6° |
| 数学单元测试覆盖 | ✅ 116 个测试覆盖：pixel↔spherical↔ray 双向、6 个已知方向、边界 wrap、极点、旋转矩阵正交性/行列式/逆、R_ypr 与 spherical 一致、FOV 透视/equirect/立体角 |

## Remaining Risks

- 极点（pitch=±π/2）处 yaw 不可逆（球面坐标奇点），但 `ray_to_spherical` 用 `atan2(0, 0)=0` 给出稳定结果，不抛异常。
- equirect 子区域角范围用线性映射，在极点附近失真（cos(pitch)→0）；立体角用球面积分更准确。
- 未实现"从 equirect 抽取透视视图"（perspective view extraction）。这是 M011 VO 基线的需求，可在 M011 时基于本模块的 `pixel_to_ray` + `rotation_ypr` 实现。
- 未处理 fisheye / 双鱼眼镜头模型（Osmo 360 原始输出可能是双鱼眼，需先 stitch 成 equirect；本模块假设输入已是 equirect）。

## Next Missions

- **M011 Visual Odometry 基线**：依赖 M006 标定 K + M007 投影模型。VO 可在 equirect 上直接做（用 `pixel_to_ray` 把像素映射到球面射线），或用 `rotation_ypr` 抽取虚拟透视视图做特征匹配。
- **M009 深度估计基线**：全景深度模型（如 PanoShape）可直接消费 equirect 像素；透视深度模型需先抽视图。
- **M015 点云生成**：equirect + depth → 3D 点云时，用 `pixel_to_ray` 把每个像素的深度反投影成 3D 点。
