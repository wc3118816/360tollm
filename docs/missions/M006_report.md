# MISSION_REPORT — M006 相机内参标定

## Goal

得到投影模型所需内参与畸变参数。AC：提供重投影误差；误差阈值可配置；标定参数可被运行时加载。

## Assumptions

- M005 已建立 DatasetIngestor + Frame API + `intrinsics.json` 格式约定
- M006 填充标定结果，输出 Calibration JSON，与 M005 `intrinsics.json` 格式兼容
- 无真实相机阶段，用合成棋盘格图像验证标定全流程
- OpenCV `calibrateCamera` 作为标定求解器（成熟、稳定、零额外依赖成本）

## Files Changed

新增：
- `modules/camera/calibration.py`（`CameraCalibration` 数据结构 + to_dict/from_dict/to_frame_intrinsics）
- `modules/camera/calibrator.py`（`Calibrator` 棋盘格标定器 + `CalibrationError`）
- `modules/camera/io.py`（`save_calibration` / `load_calibration` / `load_calibration_to_frame_intrinsics`）
- `modules/camera/__init__.py`（子包导出）
- `scripts/make_chessboard_images.py`（生成 20 张合成棋盘格图像）
- `tests/unit/test_calibration.py`（7 单测：构造/校验/序列化/Frame 格式）
- `tests/unit/test_calibration_io.py`（4 单测：save/load 往返/缺失文件/非法 JSON/建父目录）
- `tests/integration/test_calibrator_e2e.py`（6 集成测试：标定成功/fx fy 合理/往返/阈值失败/图像不足/ndarray 输入）

修改：
- `modules/config/settings.py`：新增 `CalibrationCfg`（max_reproj_error / chessboard_cols / chessboard_rows / square_size / min_valid_images）
- `configs/default.yaml`：新增 `calibration` 段
- `README.md`：状态表更新 M006 = DONE

## Implementation

### 1. CameraCalibration 数据结构（calibration.py）
```python
@dataclass(slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray  # (3,3) [[fx,0,cx],[0,fy,cy],[0,0,1]]
    dist_coeffs: np.ndarray    # (5,) [k1,k2,p1,p2,k3]
    reproj_error: float        # 平均重投影误差 (像素)
    width: int
    height: int
    ...
```
- `__post_init__`：camera_matrix 强制 (3,3) float64；dist_coeffs 不足 5 补零；reproj_error >= 0
- `fx/fy/cx/cy` 属性直接索引 K 矩阵
- `to_dict()`：JSON 兼容（含 fx/fy/cx/cy 冗余字段，便于人读）
- `to_frame_intrinsics()`：与 M005 `intrinsics.json` 格式一致，可直接注入 `Frame.intrinsics`
- `from_dict()`：反序列化，字段缺失用默认值

### 2. Calibrator 棋盘格标定器（calibrator.py）
```python
cal = Calibrator(cfg)
result = cal.calibrate(image_paths, source="chessboard_9x6")
```
流程：
1. 逐图加载灰度 → `cv2.findChessboardCorners`（ADAPTIVE_THRESH + NORMALIZE_IMAGE）
2. `cv2.cornerSubPix` 亚像素精化（5x5 窗口，30 次迭代，1e-3 精度）
3. `cv2.calibrateCamera` 求解 K + D + rvecs + tvecs
4. `_compute_reproj_error`：对每图 `cv2.projectPoints` 投影 → 与检测角点比 → numpy 欧氏距离均值
5. reproj_error > `cfg.max_reproj_error` → 抛 `CalibrationError`
6. 有效图像 < `cfg.min_valid_images` → 抛 `CalibrationError`

关键修复（开发期）：
- `objectPoints` 必须为 `np.float32`（OpenCV 要求，float64 会报错）
- 重投影误差用 numpy 而非 `cv2.norm`（避免 CV_32FC2 vs CV_32FC1 类型不匹配）

### 3. 运行时加载（io.py）
```python
save_calibration(result, "configs/calib.json")    # 写 JSON
cal = load_calibration("configs/calib.json")       # 读 JSON → CameraCalibration
intrinsics = load_calibration_to_frame_intrinsics("configs/calib.json")  # 直接 → Frame.intrinsics
```
- JSON 格式与 M005 `intrinsics.json` 兼容
- `load_calibration_to_frame_intrinsics`：一行完成 "加载 → 注入 Frame"

### 4. 配置项（settings.py + default.yaml）
```yaml
calibration:
  max_reproj_error: 1.0       # 像素, > 该阈值视为标定失败
  chessboard_cols: 9          # 棋盘格内角点列数
  chessboard_rows: 6          # 棋盘格内角点行数
  square_size: 0.025          # 格子边长 (米)
  min_valid_images: 5         # 最少有效图像数
```
- `CalibrationCfg` 可通过环境变量覆盖：`TOLLM_CALIBRATION__MAX_REPROJ_ERROR=0.5`
- 阈值可配 → AC "误差阈值可配置" 满足

## Tests

```bash
uv run pytest tests/unit/test_calibration.py tests/unit/test_calibration_io.py \
  tests/integration/test_calibrator_e2e.py -v
# 18 passed in 1.90s
uv run pytest -q   # 全量 123 passed
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖

**test_calibration.py (7 单测)**：
- 构造 + fx/fy/cx/cy 属性
- camera_matrix 形状校验（非 (3,3) 抛 ValueError）
- reproj_error 负值拒绝
- dist_coeffs 不足 5 自动补零
- to_dict / from_dict 往返一致
- to_frame_intrinsics 格式正确
- to_dict 输出 JSON 可序列化

**test_calibration_io.py (4 单测)**：
- save → load 往返字段一致
- 加载缺失文件抛 FileNotFoundError
- 加载非法 JSON 抛 ValueError
- save 自动创建父目录
- load_to_frame_intrinsics 输出格式

**test_calibrator_e2e.py (6 集成测试)**：
- 标定成功，reproj_error > 0 且 < 阈值
- fx/fy/cx/cy 均为正数
- save → load 往返一致
- 极低阈值（0.001px）触发 CalibrationError
- min_valid_images=100 触发图像不足错误
- ndarray 输入（不经文件路径）可标定

## Metrics

| 指标 | 值 |
|---|---|
| M006 测试数 | 18 (11 unit + 6 integration + 1 io roundtrip) |
| 全量测试数 | 123 (100%) |
| M006 测试耗时 | 1.90s |
| 全量测试耗时 | 39.79s |
| Lint | All checks passed |
| Format | 8 files already formatted |
| 代码行数 | ~220 (calibration.py 130 + calibrator.py 224 + io.py 87) + ~280 (测试) |

## AC 对照

| AC | 实测 |
|---|---|
| 提供重投影误差 | ✅ `CameraCalibration.reproj_error` 字段；`_compute_reproj_error` 用 `cv2.projectPoints` + numpy 欧氏距离计算 |
| 误差阈值可配置 | ✅ `CalibrationCfg.max_reproj_error`（YAML / 环境变量）；超阈值抛 `CalibrationError`；测试 `test_low_threshold_raises_error` 验证 |
| 标定参数可被运行时加载 | ✅ `load_calibration(path)` → `CameraCalibration`；`load_calibration_to_frame_intrinsics(path)` → `Frame.intrinsics` 格式；测试 `test_save_load_roundtrip` 验证 |

## Remaining Risks

- 合成棋盘格图像用仿射变换（±15°）模拟视角变化，与真实相机标定（多角度、有透视畸变）有差距。真实设备接入后需用真实棋盘格图像重新标定验证。
- OpenCV 5 参数畸变模型（k1,k2,p1,p2,k3）对广角/鱼眼镜头可能不足。全景相机（Osmo 360）的 equirectangular 模型不走此标定路径，M007 单独处理。
- `make_chessboard_images.py` 生成的是灰度棋盘格，真实标定建议用彩色棋盘格 + 良好光照。
- 当前标定结果未自动注入 Frame.intrinsics（需手动调 `load_calibration_to_frame_intrinsics`）。M011+ 阶段可加自动加载逻辑。

## Next Missions

- **M007 全景投影模型**：equirectangular ↔ spherical ↔ camera-ray 投影库。M006 的标定结果用于非全景相机的透视投影；全景相机走 M007 的球面模型。
- **M011 Visual Odometry 基线**：依赖 M006 标定结果做帧间位姿估计（PnP / 对极几何需要内参 K）。
- **M009 深度估计基线**：单目深度模型通常不需精确内参，但 M015 点云生成需要 K 做反投影。
