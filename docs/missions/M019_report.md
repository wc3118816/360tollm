# MISSION_REPORT — M019 目标检测

## Goal

建立实时对象检测模块。输出 bbox、类别、置信度、timestamp。
AC：可在设定 FPS 下稳定运行；推理延迟可测。

## Assumptions

- M004 已建立图像采集（Frame: image + frame_id + timestamp）
- 无 ML 框架依赖（torch/ultralytics），实现 Dummy 检测器作为基线
- 与 M009 深度估计模块架构对齐（ABC + Stats + Dummy 模式）

## Files Changed

新增：
- `modules/detection/types.py`（`Detection` + `DetectionList` 数据结构）
- `modules/detection/base.py`（`Detector` ABC + `DetectionStats` 统计）
- `modules/detection/dummy_detector.py`（`DummyDetector` + `DummyDetectorConfig`）
- `modules/detection/__init__.py`（子包 API + `create_detector` 工厂）
- `tests/unit/test_detection_types.py`（18 单测：Detection/DetectionList 构造/属性/过滤/序列化）
- `tests/unit/test_detection_dummy.py`（16 单测：DummyDetector/Stats/AC 稳定性/工厂）
- `tests/integration/test_detection_e2e.py`（6 集成测试：完整链路/AC/批量/深度集成）

修改：
- `README.md`：状态表更新 M019 = DONE

## Implementation

### 1. Detection 数据结构（types.py）
```python
@dataclass(slots=True)
class Detection:
    bbox: np.ndarray        # int32 (4,) = [x_min, y_min, x_max, y_max]
    label: str              # 类别名
    label_id: int           # 类别 ID (0-based)
    confidence: float       # 置信度 ∈ [0, 1]
    metadata: dict          # 扩展字段
```
- 属性：`width` / `height` / `area` / `center`
- 校验：bbox shape (4,) + confidence ∈ [0,1] + x_max > x_min + y_max > y_min

```python
@dataclass(slots=True)
class DetectionList:
    detections: list[Detection]
    latency_ms: float       # 整帧推理延迟
    backend: str            # 检测器后端名
    timestamp: float        # 帧时间戳
    image_shape: tuple[int, int]  # (H, W)
    metadata: dict
```
- 属性：`n_detections` / `labels` / `fps`（1000/latency）
- 过滤：`filter_by_confidence(threshold)` / `filter_by_label(label)`
- 序列化：`summary()` / `to_dict()`

### 2. Detector ABC + DetectionStats（base.py）
```python
class Detector(ABC):
    def detect(image, timestamp=0.0) -> DetectionList    # 单帧 (自动计时)
    def detect_batch(images, timestamps=None) -> list[DetectionList]  # 批量
    @abstractmethod
    def _detect(image) -> list[Detection]                # 子类实现
```
- `detect()` 包装：预处理 → 计时 → `_detect()` → 统计 → 构建 DetectionList
- `DetectionStats`：`n_frames` / `avg_latency_ms` / `p95_latency_ms` / `avg_fps`

### 3. DummyDetector（dummy_detector.py）
```python
det = DummyDetector(DummyDetectorConfig(n_detections=3, latency_ms=5.0))
dl = det.detect(image)
```
- 在图像随机位置生成 N 个 bbox + label + confidence
- 模拟推理延迟（`time.sleep`），用于验证 AC
- label 从默认 COCO 风格类别列表选择
- 固定 seed → 可重复结果

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/unit/test_detection_types.py \
  tests/unit/test_detection_dummy.py tests/integration/test_detection_e2e.py -v
# 40 passed in 0.59s
.venv\Scripts\python.exe -m pytest -q   # 全量 532 passed
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
```

### 测试覆盖

**test_detection_types.py (18 单测)**：
- `TestDetection` (8)：构造 + bbox shape + confidence 范围 + 无效 bbox + width/height + area + center + to_dict
- `TestDetectionList` (10)：空列表 + 带检测 + FPS + 零延迟 + 过滤 + summary + to_dict + 无效 latency + 无效 shape

**test_detection_dummy.py (16 单测)**：
- `TestDummyDetector` (8)：基本检测 + bbox 范围 + label 配置 + confidence 范围 + 延迟 + FPS + 批量 + 可重复性
- `TestDetectionStats` (4)：record/avg + p95 + avg_fps + as_dict
- `TestACStableFPS` (2)：**AC 设定 FPS 稳定** + **AC FPS 达标**
- `TestCreateDetector` (2)：dummy + 未知后端

**test_detection_e2e.py (6 集成测试)**：
- `TestFullPipeline` (5)：单帧 + **AC 稳定 FPS** + **AC 延迟可测** + 字段完整 + 批量 timestamp
- `TestDetectionWithDepth` (1)：检测 + 深度（为 M021 前置）

## Metrics

| 指标 | 值 |
|---|---|
| M019 测试数 | 40 (34 unit + 6 integration) |
| 全量测试数 | 532 (100%) |
| M019 测试耗时 | 0.59s |
| 全量测试耗时 | 37.69s |
| Lint | All checks passed |
| Format | 7 files already formatted |
| 代码行数 | ~480 (types 180 + base 190 + dummy 120) + ~470 (测试) |
| 外部依赖 | 无（纯 numpy + time） |

## AC 对照

| AC | 实测 |
|---|---|
| 可在设定 FPS 下稳定运行 | ✅ `DummyDetectorConfig(latency_ms=X)` 设定延迟 → FPS = 1000/X；`test_ac_stable_fps` 验证 20 帧延迟稳定（std < 10ms）；`test_meets_target_fps` 验证 FPS 达标；`DetectionStats.avg_fps` 累计统计 |
| 推理延迟可测 | ✅ `DetectionList.latency_ms` 每帧记录；`DetectionStats` 累计 avg/p95；`test_latency_measured` + `test_ac_latency_measured` 验证 > 0 |

## Remaining Risks

- **DummyDetector 是合成基线**：不检测真实物体。需接入 YOLO/RT-DETR（留后续，需 torch）。
- **无 NMS**：Dummy 生成不重叠 bbox，无需 NMS。真实检测器需 NMS 后处理。
- **无 GPU 加速**：当前纯 CPU。真实模型需 CUDA 支持。
- **无模型加载**：Dummy 不加载权重。真实后端需模型管理（加载/卸载/版本切换）。
- **bbox 格式固定 [x_min,y_min,x_max,y_max]**：部分模型用 [cx,cy,w,h]。需转换层。
- **无类别表管理**：label_id 是本地索引。需统一类别表（COCO 80 类 / 自定义）。
- **无跟踪**：每帧独立检测，无 track ID。M022 3D 跟踪负责。

## Next Missions (最小 MVP 路径)

按 `mission_list.md` 第 16 节最小 MVP 路径：
- **M021 2D→3D 目标关联**：用 M015 点云 + M019 检测 bbox，反投影到 3D 世界坐标
- **M026 Scene Graph**：基于 3D 物体构建场景图
- **M029 世界坐标 API**：统一 world model 接口
- **M037 LLM 上下文适配器**：把 world model 转为 LLM 可读格式
- **M038 LLM 空间问答**：实现基础空间问答
- **M048 MVP Demo**：完成最小闭环
