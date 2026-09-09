# MISSION_REPORT — M009 深度估计基线

## Goal

建立单目/全景图深度估计基线。输出 Depth Map。AC：可对离线数据批量推理；输出深度图与置信度；记录推理延迟。

## Assumptions

- M004 已建立 Frame 对象（含 `depth` 字段，float32 (H,W) 米，此前恒为 None）
- M005 已建立离线数据集回放（DatasetIngestor），可作为批量推理的输入源
- 本机无 GPU 或 GPU VRAM 有限，MVP 阶段优先 CPU 可跑的轻量基线
- MiDaS（Intel 开源单目深度模型）作为真实基线候选，但首次加载需联网下载权重
- 无网络/无 torch 环境下需要可测试的合成基线

## Files Changed

新增：
- `modules/depth/types.py`（`DepthMap` 数据结构：depth + confidence + latency + summary）
- `modules/depth/base.py`（`DepthEstimator` ABC + `DepthStats` + 预处理 + 计时）
- `modules/depth/dummy_estimator.py`（`DummyDepthEstimator` 合成基线，无 ML 依赖）
- `modules/depth/midas_estimator.py`（`MiDaSDepthEstimator` 真实基线，torch.hub lazy import）
- `modules/depth/__init__.py`（子包 API + `create_depth_estimator` 工厂）
- `tests/unit/test_depth_types.py`（11 单测：构造/校验/截断/序列化）
- `tests/unit/test_depth_estimator.py`（25 单测：Dummy/预处理/批量/统计/工厂）
- `tests/integration/test_depth_e2e.py`（9 集成测试：Frame→DepthMap/AC 合规/统计）

修改：
- `modules/config/settings.py`：`DepthCfg` 加 `near` / `far` 字段 + docstring
- `configs/default.yaml`：`depth.backend` 默认改为 `dummy`，加 `near` / `far`
- `README.md`：状态表更新 M009 = DONE

## Implementation

### 1. DepthMap 数据结构（types.py）
```python
@dataclass(slots=True)
class DepthMap:
    depth: np.ndarray         # float32 (H,W) 米, 0=无效
    confidence: np.ndarray   # float32 (H,W) [0,1]
    latency_ms: float        # 推理延迟 (毫秒) — AC 要求
    backend: str             # dummy / midas / metric3d / zoe / bifuze
    width: int               # 自动从 depth 推断
    height: int
    metadata: dict           # 模型版本 / device / near / far 等
```
- `__post_init__`：类型强转 float32；形状校验（depth==confidence，2D）；confidence 截断 [0,1]
- `summary()`：统计摘要（valid_pixels / depth_min/max/mean / conf_mean），用于日志/JSON

### 2. DepthEstimator ABC + 统计（base.py）
```python
est = DummyDepthEstimator(device="cpu", max_resolution=1024)
dm = est.estimate(image)              # 单帧 (自动计时 + 统计)
dms = est.estimate_batch([img1, img2]) # 批量 (AC: 可对离线数据批量推理)
est.stats.avg_latency_ms              # 平均延迟
est.stats.p95_latency_ms              # P95 延迟
```
- `estimate()` 包装 `_estimate()`：自动 `time.perf_counter()` 计时 → 写入 `dm.latency_ms` + `DepthStats.record()`
- `_preprocess()`：灰度→RGB；超过 `max_resolution` 自动下采样（最近邻）
- `estimate_batch()`：循环调 `estimate()`，每帧加 `batch_index` metadata
- `DepthStats`：n_frames / total_latency / avg / p95（保留最近 1000 条）

### 3. DummyDepthEstimator 合成基线（dummy_estimator.py）
```python
est = DummyDepthEstimator(near=0.5, far=10.0)
dm = est.estimate(image)
# depth = near + (far-near) * (0.5*radial_dist + 0.5*brightness_inv)
```
- 确定性输出（相同输入 → 相同输出，便于测试）
- 中心近（深度小）、边缘远；亮区域近、暗区域远
- 置信度：中心 1.0、边缘 0.2
- **无任何 ML 依赖**，纯 numpy

### 4. MiDaSDepthEstimator 真实基线（midas_estimator.py）
```python
est = MiDaSDepthEstimator(device="cpu", model_type="DPT_Large", near=0.5, far=20.0)
est.warmup()  # 预加载模型
dm = est.estimate(image)
```
- lazy import torch（未安装时 `_TORCH_AVAILABLE=False`，工厂抛 RuntimeError 而非 ImportError）
- `torch.hub.load("isl-org/MiDaS", "DPT_Large")` 首次加载需网络
- 输出相对深度（越近越大）→ 归一化 [0,1] → 反转 → 线性映射到 [near, far] 米
- 置信度暂用常数 1.0（MiDaS 不输出置信度，真实评测留 M010）
- 支持 `DPT_Large` / `DPT_Hybrid` / `MiDaS_small` 三种变体

### 5. 工厂 + 配置（__init__.py + settings.py）
```python
est = create_depth_estimator()  # 从 settings.depth 读
est = create_depth_estimator(DepthCfg(backend="dummy"))
```
- backend 映射：none/dummy → Dummy；midas/midas_small/midas_large → MiDaS
- 未知 backend → ValueError；torch 未装 → RuntimeError（带修复提示）
- `DepthCfg` 新增 `near` / `far` 字段，可通过环境变量覆盖

## Tests

```bash
uv run pytest tests/unit/test_depth_types.py tests/unit/test_depth_estimator.py \
  tests/integration/test_depth_e2e.py -v
# 45 passed in 0.46s
uv run pytest -q   # 全量 284 passed
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖

**test_depth_types.py (11 单测)**：
- `TestDepthMapConstruction` (3)：基本构造 + shape 推断 + shape 属性
- `TestDepthMapValidation` (4)：形状不匹配/3D 拒绝/负 latency/类型强转
- `TestConfidenceClipping` (1)：confidence 截断 [0,1]
- `TestSerialization` (3)：to_dict / summary 有效像素 / summary 全零

**test_depth_estimator.py (25 单测)**：
- `TestDummyDepthEstimator` (8)：输出形状/深度范围/置信度范围/中心近/确定性/latency/metadata/describe
- `TestDepthEstimatorPreprocess` (4)：灰度输入/非法形状/下采样/小图不放大
- `TestBatchInference` (4)：返回 list/每帧有 depth/latency 记录/stats 累积
- `TestDepthStats` (4)：空统计/avg/p95/历史截断
- `TestFactory` (5)：从 cfg 创建/none→dummy/未知 backend/从 settings/MiDaS 无 torch 报错

**test_depth_e2e.py (9 集成测试)**：
- `TestFrameToDepthMap` (2)：Frame.image→DepthMap / DepthMap→Frame.depth 注入
- `TestBatchOfflineData` (2)：批量帧推理/延迟一致性
- `TestACCompliance` (3)：**AC 三项** — 输出深度+置信度 / 记录延迟 / 批量推理
- `TestStatsPersistence` (1)：多帧 stats 累积
- `TestFromSettings` (1)：从默认配置创建

## Metrics

| 指标 | 值 |
|---|---|
| M009 测试数 | 45 (36 unit + 9 integration) |
| 全量测试数 | 284 (100%) |
| M009 测试耗时 | 0.46s |
| 全量测试耗时 | 36.47s |
| Lint | All checks passed |
| Format | 8 files already formatted |
| 代码行数 | ~250 (types 90 + base 180 + dummy 75 + midas 110) + ~450 (测试) |
| 外部依赖 | Dummy 无依赖；MiDaS 需 torch (lazy import) |

## AC 对照

| AC | 实测 |
|---|---|
| 可对离线数据批量推理 | ✅ `DepthEstimator.estimate_batch(images)`；测试 `test_batch_inference` 验证 10 帧批量；Frame.image 可直接作为输入 |
| 输出深度图与置信度 | ✅ `DepthMap.depth` (float32 (H,W) 米) + `DepthMap.confidence` (float32 (H,W) [0,1])；测试 `test_output_depth_and_confidence` 验证形状与 dtype |
| 记录推理延迟 | ✅ `DepthMap.latency_ms` (毫秒) + `DepthStats.avg_latency_ms` / `p95_latency_ms`；测试 `test_latency_recorded` 验证 > 0 |

## Remaining Risks

- **Dummy 输出是合成深度，不反映真实几何**。仅用于流水线测试和占位。真实场景必须切到 MiDaS / Metric3D / Zoe。
- **MiDaS 是相对深度，非绝对米数**。线性映射到 [near, far] 是启发式，不精确。Metric3D / Zoe 可输出绝对深度，留 M010 评测后切换。
- **MiDaS 不输出置信度**。当前用常数 1.0，真实置信度需 M010 通过多次推理方差或多模型集成获得。
- **CPU 上 MiDaS DPT_Large 推理慢**（单帧可能数秒）。MVP 阶段建议用 `MiDaS_small` 或 `max_resolution=512` 降负载。
- **全景图（equirectangular）深度**：MiDaS 是为透视图训练的，对 equirect 有畸变。全景专用模型（如 PanoShape）留后续。
- **首次加载 MiDaS 需联网**下载权重（~1GB）。离线环境需预下载或用 Dummy。

## Next Missions

- **M010 深度质量评测**：建立 Abs Rel / RMSE / δ accuracy benchmark，量化 Dummy vs MiDaS 误差，决定是否升级到 Metric3D。
- **M015 点云生成**：依赖 M009 深度 + M011 相机位姿。`pixel_to_ray` (M007) + `DepthMap.depth` → 3D 点云。
- **M011 Visual Odometry 基线**：M009 深度可辅助 VO（单目尺度模糊时），但 M011 主要依赖 M006 标定 + M007 投影。
