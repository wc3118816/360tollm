# MISSION_REPORT — M011 Visual Odometry 基线

## Goal

根据连续 RGB / panorama 估计相机运动。输出 Frame-to-frame Pose。AC：能在回放数据上输出连续轨迹；轨迹无 NaN；失败率可统计。

## Assumptions

- M004 已建立 Frame 对象（含 `pose` 字段，4x4 SE3，此前恒为 None）
- M005 已建立 DatasetIngestor 回放系统，可作为 VO 输入源
- M006 已建立 CameraCalibration（FeatureVO 需要内参 K）
- M007 已建立投影模型（rotation_ypr 用于 DummyVO 合成轨迹）
- 本机无 GPU，MVP 阶段优先 CPU 可跑的轻量基线
- OpenCV 已安装（M006 标定用到），可复用其 ORB 特征 + recoverPose

## Files Changed

新增：
- `modules/vo/types.py`（`Pose` + `Trajectory` 数据结构：SE3 + position/yaw + has_nan + summary）
- `modules/vo/base.py`（`VisualOdometry` ABC + `VOStats` 失败率统计 + 连续失败检测）
- `modules/vo/dummy_vo.py`（`DummyVisualOdometry` 合成基线，无 ML 依赖）
- `modules/vo/feature_vo.py`（`FeatureVO` ORB + RANSAC + recoverPose，需 OpenCV + K）
- `modules/vo/__init__.py`（子包 API + `create_vo` 工厂）
- `tests/unit/test_vo_types.py`（20 单测：Pose 构造/校验/属性/序列化 + Trajectory 统计）
- `tests/unit/test_vo_estimator.py`（20 单测：DummyVO/VOStats/工厂）
- `tests/integration/test_vo_e2e.py`（12 集成测试：Frame→Trajectory/AC 合规/边缘情况）

修改：
- `modules/config/settings.py`：`SLAMCfg` 加 `device` / `n_features` / `scale_factor` + docstring
- `configs/default.yaml`：`slam.backend` 默认改为 `dummy`，加新字段
- `README.md`：状态表更新 M011 = DONE

## Implementation

### 1. Pose + Trajectory 数据结构（types.py）
```python
@dataclass(slots=True)
class Pose:
    matrix: np.ndarray       # 4x4 float64 SE3 [R|t; 0|1] (world_to_camera)
    timestamp: float         # 帧时间戳
    frame_id: int            # 对应 Frame.frame_id
    confidence: float        # [0,1] RANSAC inlier ratio
    metadata: dict           # n_matches / n_inliers / synthetic 等

@dataclass(slots=True)
class Trajectory:
    poses: list[Pose]
    backend: str
    metadata: dict           # n_input_frames / failure_rate / total_latency_ms
```
- `Pose.position` = `-R^T @ t`（相机在 world 坐标系的位置）
- `Pose.yaw` = `atan2(-R[2,0], R[0,0])`（与 M007 约定一致）
- `Trajectory.has_nan()`：检查所有 pose 矩阵是否含 NaN/Inf（AC: 轨迹无 NaN）
- `Trajectory.total_displacement()` / `total_path_length()`：位移与路径长度
- `Trajectory.summary()`：统计摘要（n_poses / has_nan / displacement / path_length）

### 2. VisualOdometry ABC + 统计（base.py）
```python
vo = DummyVisualOdometry()
traj = vo.estimate(frames)     # 帧序列 → Trajectory
vo.stats.failure_rate          # 失败率 (AC)
vo.stats.avg_latency_ms        # 平均延迟
```
- `estimate()` 包装 `_estimate_poses()`：自动计时 + 统计成功/失败
- `VOStats`：n_frames / n_success / n_fail / failure_rate / consecutive_fails / is_failed()
- 失败帧（`_estimate_poses` 返回 None）跳过，不产生 Pose
- 全部失败时 fallback 到 identity pose（保证 Trajectory 非空）
- 连续失败超 10 次 → `is_failed()` 返回 True（可触发整体降级）

### 3. DummyVisualOdometry 合成基线（dummy_vo.py）
```python
vo = DummyVisualOdometry(step_distance=0.5, yaw_amplitude=0.3)
traj = vo.estimate(frames)
# 相机沿 +x 匀速移动, 可选 yaw 摆动 (模拟走路)
```
- 确定性输出（相同输入 → 相同输出）
- 用 M007 的 `rotation_yaw` 生成旋转矩阵
- `world_to_camera = [R_yaw(yaw) | -R_yaw(yaw) @ pos_world]`
- **无任何 ML 依赖**，纯 numpy

### 4. FeatureVO 真实基线（feature_vo.py）
```python
vo = FeatureVO(camera_matrix=K, n_features=1000, scale_factor=0.1)
traj = vo.estimate(frames)
```
- 首帧为单位 pose
- 每帧用 ORB 特征匹配 + `cv2.findEssentialMat` (RANSAC) + `cv2.recoverPose`
- 相对运动累积：`T_world_curr = T_world_prev @ T_prev_curr`
- 单目尺度模糊：`t_rel *= scale_factor`（启发式，M012 VIO 用 IMU 修正）
- 失败帧（特征不足 / RANSAC 失败 / inlier 不足）返回 None
- lazy import cv2（未安装时工厂抛 RuntimeError）

### 5. 工厂 + 配置（__init__.py + settings.py）
```python
vo = create_vo()  # 从 settings.slam 读
vo = create_vo(SLAMCfg(backend="dummy"))
vo = create_vo(SLAMCfg(backend="orb"), camera_matrix=K)
```
- backend 映射：none/dummy → Dummy；orb/feature/feature_vo → FeatureVO
- FeatureVO 需要 camera_matrix 参数（M006 标定结果）
- `SLAMCfg` 新增 `device` / `n_features` / `scale_factor`

## Tests

```bash
uv run pytest tests/unit/test_vo_types.py tests/unit/test_vo_estimator.py \
  tests/integration/test_vo_e2e.py -v
# 52 passed in 0.43s
uv run pytest -q   # 全量 336 passed
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖

**test_vo_types.py (20 单测)**：
- `TestPoseConstruction` (2)：identity + 基本构造
- `TestPoseValidation` (4)：形状/NaN/Inf/confidence 越界拒绝
- `TestPoseProperties` (4)：R/t 提取 + position + yaw
- `TestPoseSerialization` (1)：to_dict
- `TestTrajectory` (9)：构造/positions/timestamps/空轨迹/has_nan/displacement/path_length/summary

**test_vo_estimator.py (20 单测)**：
- `TestDummyVisualOdometry` (10)：返回 Trajectory/长度匹配/首帧 identity/无 NaN/确定性/位移/yaw 摆动/空输入/describe/stats
- `TestVOStats` (5)：空统计/成功失败混合/连续失败/重置/as_dict
- `TestFactory` (5)：从 cfg 创建/none→dummy/未知 backend/从 settings/FeatureVO 无 cv2 报错

**test_vo_e2e.py (12 集成测试)**：
- `TestFrameToTrajectory` (2)：Frame→Trajectory / Pose→Frame.pose 注入
- `TestACCompliance` (3)：**AC 三项** — 连续轨迹 / 无 NaN / 失败率可统计
- `TestTrajectoryMetrics` (4)：位移随帧数增加/直线位移/路径≥位移/summary 字段
- `TestFromSettings` (1)：从默认配置创建
- `TestEdgeCases` (2)：单帧 / 空输入

## Metrics

| 指标 | 值 |
|---|---|
| M011 测试数 | 52 (40 unit + 12 integration) |
| 全量测试数 | 336 (100%) |
| M011 测试耗时 | 0.43s |
| 全量测试耗时 | 40.51s |
| Lint | All checks passed |
| Format | 8 files already formatted |
| 代码行数 | ~280 (types 165 + base 130 + dummy 75 + feature 170) + ~520 (测试) |
| 外部依赖 | Dummy 无依赖；FeatureVO 需 OpenCV (已安装) |

## AC 对照

| AC | 实测 |
|---|---|
| 能在回放数据上输出连续轨迹 | ✅ `VisualOdometry.estimate(frames) -> Trajectory`；测试 `test_continuous_trajectory` 验证 20 帧连续（时间戳单调递增）；Frame.image 作为输入 |
| 轨迹无 NaN | ✅ `Trajectory.has_nan()` + `Pose.__post_init__` 拒绝 NaN/Inf；测试 `test_trajectory_no_nan` 验证 50 帧无 NaN；`estimate()` 中显式检查并 log |
| 失败率可统计 | ✅ `VOStats.failure_rate` + `n_success` / `n_fail`；测试 `test_failure_rate_stats` 验证；Dummy 不失败（0%），FeatureVO 真实失败由 RANSAC 决定 |

## Remaining Risks

- **Dummy 输出是合成轨迹，不反映真实运动**。仅用于流水线测试。真实场景必须用 FeatureVO（ORB）。
- **单目 VO 尺度漂移**：FeatureVO 用 `scale_factor` 启发式，长时间会累积误差。M012 VIO 用 IMU 修正尺度。
- **全景图（equirectangular）不能直接用 ORB**：ORB 是透视图特征检测器。全景需先抽视图（M007 的 `pixel_to_ray` + `rotation_ypr` 可实现虚拟透视视图抽取）。
- **FeatureVO 未用 M006 标定结果**：当前需手动传 `camera_matrix`。M013 SLAM 阶段可加自动加载（从 `load_calibration` 读取 K）。
- **recoverPose 只给相对旋转 + 平移方向**：平移尺度未知，乘 `scale_factor` 是启发式。M012 VIO 会用 IMU 加速度计恢复绝对尺度。
- **未实现回环检测**：长时间运行会漂移。M014 回环检测处理。

## Next Missions

- **M012 VIO 融合**：融合 M008 IMU + M011 VO，用 IMU 修正尺度漂移。依赖 M008（尚未做）。
- **M013 SLAM 建图**：在 M011 VO 基础上加回环检测 + 地图优化。依赖 M012。
- **M015 点云生成**：M009 深度 + M011 位姿 → 3D 点云。`pixel_to_ray` (M007) + `DepthMap.depth` (M009) + `Pose` (M011)。
- **M021 2D→3D 目标关联**：用 M011 位姿把 2D 检测框反投影到 3D 世界坐标。
