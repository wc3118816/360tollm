# 360ToLLM — 360° Camera Spatial Intelligence / 3D World Model for LLM

> **相机连续观察现实世界后，能否形成结构化、可查询、带空间和时间关系的 World Model，并让 LLM 基于这个 World Model 做 grounded reasoning？**

将 DJI Osmo 360 等全景相机的连续视频流，转换为结构化、可查询、带空间和时间关系的 **World Model**，让 LLM 基于该 World Model 做 grounded reasoning，实现空间智能问答。

---

## 项目属性

| 属性 | 值 |
|---|---|
| **项目类型** | 空间智能 / 3D World Model / LLM Grounding |
| **编程语言** | Python ≥ 3.11 |
| **依赖管理** | [uv](https://github.com/astral-sh/uv) |
| **测试框架** | pytest + pytest-cov |
| **Lint/Format** | ruff |
| **配置** | pydantic-settings + YAML |
| **日志** | structlog |
| **外部依赖** | 纯 numpy（核心模块零外部依赖） |
| **测试总数** | 765 (100% 通过) |
| **License** | MIT |

---

## 创新点

### 1. 端到端 3D 空间理解闭环

从 360° 全景相机视频流到 LLM 空间问答的完整闭环，覆盖：

```
Osmo 360 → Video → Frame → Object Detection → Depth
→ Camera Pose → 3D Object → World Model → LLM → 用户问题
```

**创新**：不同于传统 LLM 只处理文本，本项目让 LLM 获得物理世界的 3D 空间感知能力。

### 2. 统一 World Model 架构

- **10 字段 JSON Schema**：scene_id / coordinate_system / camera_pose / objects / relations / regions / occupancy / memory / confidence / timestamp
- **SemVer 版本管理**：向后兼容策略明确，支持跨版本数据迁移
- **SI 单位统一**：距离（米）、角度（弧度），坐标变换双向可验证

**创新**：为 LLM 提供标准化、可序列化、可查询的空间世界模型。

### 3. 空间记忆与时间衰减

- **跨帧物体关联**：同 label + 距离阈值 → 同一 tracked_id
- **观测历史**：TrackedObject 保存完整观测序列 + 时间戳
- **"刚才那个物体"**：SpatialMemory.where_is(tracked_id) 支持时间窗口查询

**创新**：LLM 可回答涉及时间维度的问题（"刚才那个人去哪了？"）。

### 4. LLM 上下文最小化（不发送整张地图）

- **8 种查询类型**：WHERE_IS / DISTANCE / DIRECTION / COUNT / LIST_OBJECTS / NEAREST / RECENT / CUSTOM
- **max_objects 限制**：同一问题不会无条件发送整张地图
- **几何事实优先**：只返回物体摘要（ID/label/position/confidence），不发送原始点云
- **保留置信度 + 时间**：每个证据字段含 confidence + last_seen

**创新**：在保证 LLM 获得足够上下文的同时，控制 token 消耗。

### 5. 结构化证据引用

- **Answer.evidence**：答案直接引用 World Model 中的字段（object_id / position / confidence / distance）
- **可验证性**：LLM 答案可追溯到底层 World Model 数据

**创新**：LLM 空间问答不再是"黑盒"，每个答案都有结构化证据支撑。

### 6. 全景投影模型

- **equirect ↔ spherical ↔ camera-ray** 三向变换
- **R_ypr**（yaw-pitch-roll）旋转矩阵
- **FOV 裁剪**：从全景图像提取 perspective 视图

**创新**：支持从 360° 全景图像中提取任意视角的 3D 信息。

### 7. 占用栅格 + 射线投射

- **三态栅格**：UNKNOWN / FREE / OCCUPIED
- **3D Bresenham 射线投射**：点云→栅格标记 free
- **时间衰减**：confidence 衰减 + 低于阈值→UNKNOWN

**创新**：为 LLM 提供可查询的空间占用状态。

### 8. 零外部依赖核心

核心模块（点云、网格、占用栅格、序列化、QA）**零外部依赖**，纯 numpy 实现：

| 模块 | 依赖 |
|---|---|
| pointcloud (法向量+滤波+优化) | numpy |
| mesh (三角化+glTF/PLY/OBJ) | numpy + struct + json |
| occupancy (射线投射+衰减) | numpy |
| serialization (JSON Schema) | json + time |
| spatial_qa (Parser+QA) | re + numpy |
| llm_context (Adapter) | numpy |

**创新**：MVP 阶段无需安装重型依赖（Open3D/PCL/PyTorch3D），降低部署门槛。

---

## 架构概览

```
modules/
├── config/          # pydantic-settings 配置管理
├── logging/         # structlog 日志
├── ingest/          # RTMP/File 视频流接入 (M003-M005)
├── camera/          # 相机标定 (M006)
├── panorama/        # 全景投影模型 (M007)
├── depth/           # 深度估计 Dummy/MiDaS (M009)
├── vo/              # Visual Odometry Dummy/Feature (M011)
├── pointcloud/      # 点云生成+优化 (M015-M016)
├── mesh/            # Mesh 重建 glTF/PLY/OBJ (M017)
├── occupancy/       # 占用栅格+射线投射 (M018)
├── detection/       # 目标检测 Dummy (M019)
├── objects/         # 2D→3D 关联 (M021)
├── scene_graph/     # 语义场景图 (M026)
├── world_model/     # World Model API (M029)
├── spatial_memory/  # 空间记忆+跨帧跟踪 (M031)
├── serialization/   # JSON Schema+版本管理 (M036)
├── llm_context/     # LLM 上下文适配器 (M037)
├── spatial_qa/      # 空间问答 Agent (M038)
├── demo/            # MVP Demo (M048)
└── pipeline/        # 帧管线 (M004)
```

---

## Mission 完成状态

### 最小 MVP 路径（14 个 Mission 全部完成）

| Mission | 状态 | 核心功能 | 测试数 |
|---|---|---|---|
| M001 项目工程骨架 | DONE | 配置+日志+CI | — |
| M002 设备输入方案 | DONE | 离线数据+RTMP方案 | — |
| M003 RTMP/视频流接入 | DONE | Ingestor ABC + RTMP + File | 49 |
| M004 视频解码与抽帧 | DONE | Frame.fps + 丢帧统计 | 68 |
| M005 数据集与回放系统 | DONE | DatasetIngestor + TimeSync | 105 |
| M006 相机内参标定 | DONE | CameraCalibration + Calibrator | 123 |
| M007 全景投影模型 | DONE | equirect ↔ spherical ↔ ray | 239 |
| M009 深度估计基线 | DONE | DepthMap + Dummy/MiDaS | 284 |
| M011 Visual Odometry | DONE | Pose/Trajectory + Dummy/Feature | 336 |
| M015 点云生成 | DONE | PointCloud + Generator + PLY | 379 |
| M016 点云优化 | DONE | 法向量 + 统计/半径滤波 | 411 |
| M017 Mesh 重建 | DONE | TriangleMesh + glTF/PLY/OBJ | 455 |
| M018 Occupancy Map | DONE | 三态栅格 + 射线投射 + 衰减 | 492 |
| M019 目标检测 | DONE | Detection + DummyDetector | 532 |
| M021 2D→3D 关联 | DONE | Object3D + ObjectAssociator | 561 |
| M026 Scene Graph | DONE | SceneNode+Edge+Graph+Builder | 597 |
| M029 World Model API | DONE | 坐标变换+物体查询+SI单位 | 629 |
| M031 Spatial Memory | DONE | TrackedObject+跨帧关联 | 662 |
| M036 World Model 序列化 | DONE | 10字段+版本+向后兼容 | 696 |
| M037 LLM 上下文适配器 | DONE | 8查询类型+max_objects限制 | 729 |
| M038 LLM 空间问答 | DONE | Parser+Adapter+AnswerGenerator | 751 |
| M048 MVP Demo | DONE | 端到端闭环+4问题+证据 | 765 |

**全量 765 测试 100% 通过**

---

## MVP Demo 验证

### 4 个示例问题（AC：闭环 + 结构化证据）

| # | 问题 | 查询类型 | 证据字段 |
|---|---|---|---|
| 1 | "我前面有什么？" | LIST_OBJECTS | object_id, label, confidence |
| 2 | "左边是什么？" | LIST_OBJECTS | object_id, label, confidence |
| 3 | "桌子离我多远？" | DISTANCE | object_id, distance, confidence |
| 4 | "刚才的人在哪里？" | RECENT | tracked_id, position, last_seen, confidence |

### 运行 Demo

```python
from modules.demo import MVPDemo

demo = MVPDemo(n_frames=5)
demo.run()            # 完整流程
demo.print_results()  # 打印结果
```

---

## 快速开始

### 1. 安装

```bash
# 需要 uv (https://docs.astral.sh/uv/)
uv sync --group dev
```

### 2. 运行测试

```bash
uv run pytest -q          # 全量 765 测试
uv run pytest -m unit     # 仅单测
```

### 3. Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

### 4. MVP Demo

```bash
uv run python -c "from modules.demo import MVPDemo; MVPDemo().print_results()"
```

---

## 工程目录

```
360tollm/
├── apps/              # 可执行入口 (replay, qa_agent)
├── modules/           # 核心模块 (22 个子包)
├── configs/           # YAML 配置 (default.yaml + local.yaml)
├── datasets/          # 数据集与录制 (内容 gitignored)
├── tests/             # unit / integration (765 测试)
├── scripts/           # 一次性脚本与工具
├── docs/              # 文档与 Mission 列表
├── .github/           # CI workflows
├── pyproject.toml     # 依赖与工具配置
└── ruff.toml          # Lint 规则
```

---

## 配置覆盖

按以下优先级（后者覆盖前者）：

1. `configs/default.yaml` (committed)
2. `configs/local.yaml` (gitignored, 自由编辑)
3. `$TOLLM_CONFIG_PATH` 指向的 yaml
4. 环境变量，前缀 `TOLLM_`，嵌套用 `__`，例：

   ```
   export TOLLM_LOG__LEVEL=DEBUG
   export TOLLM_INGEST__TARGET_FPS=5.0
   ```

---

## 技术栈

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) 依赖管理
- pydantic-settings + YAML 配置
- structlog 日志
- pytest + pytest-cov 测试
- ruff Lint/Format
- numpy 核心计算（零外部依赖）

---

## License

MIT
