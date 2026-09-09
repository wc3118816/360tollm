# Mission List — 360°相机空间智能 / 3D World Model

## 0. 文档目标

本任务列表用于将“DJI Osmo 360 + 视觉感知 + 3D空间理解 + LLM Agent”项目拆解为可由软件 Agent / Coding Agent 独立领取、执行、验证和交付的 Mission。

每个 Mission 应满足：

- 有明确输入与输出
- 有可自动验证的 Acceptance Criteria
- 尽量单一职责
- 可以通过文件、API、测试或指标判断完成与否
- 明确前置依赖
- 避免把“研究”和“编码”混成一个不可验收的大任务

---

# 1. 总体技术目标

```text
DJI Osmo 360
    ↓
Video / Image Ingestion
    ↓
RGB + IMU / Metadata
    ↓
Camera Calibration
    ↓
Projection / Panorama Processing
    ↓
Depth Estimation / Depth Sensor Integration
    ↓
Visual Odometry / VIO / SLAM
    ↓
Camera Pose
    ↓
Point Cloud / Mesh / Occupancy
    ↓
Object Detection / Segmentation
    ↓
3D Object Association / Tracking
    ↓
Semantic 3D Scene
    ↓
Scene Graph
    ↓
4D Spatial Memory
    ↓
World Model API
    ↓
LLM / Realtime Agent
    ↓
Spatial Reasoning / Planning / Dialogue
```

---

# 2. Mission 状态定义

| 状态 | 含义 |
|---|---|
| TODO | 尚未开始 |
| READY | 依赖满足，可执行 |
| RUNNING | 正在执行 |
| BLOCKED | 被外部条件阻塞 |
| REVIEW | 已完成，等待验证 |
| DONE | 已通过验收 |
| FAILED | 验收失败，需要返工 |
| CANCELLED | 不再执行 |

---

# 3. Mission 优先级

| 优先级 | 含义 |
|---|---|
| P0 | MVP 必须完成 |
| P1 | 核心能力，建议完成 |
| P2 | 增强空间理解能力 |
| P3 | 高级研究能力 |

---

# 4. Mission 总览

| ID | Mission | Priority | 依赖 | 产物 |
|---|---|---:|---|---|
| M001 | 项目工程骨架 | P0 | - | Repo / CI |
| M002 | 设备输入方案确认 | P0 | M001 | Ingestion Spec |
| M003 | RTMP/视频流接入 | P0 | M002 | Live Ingestor |
| M004 | 视频解码与抽帧 | P0 | M003 | Frame Pipeline |
| M005 | 数据集与回放系统 | P0 | M004 | Replay System |
| M006 | 相机内参标定 | P0 | M005 | Calibration |
| M007 | 全景投影模型 | P1 | M006 | Projection Library |
| M008 | IMU数据接入 | P1 | M002 | IMU Pipeline |
| M009 | 深度估计基线 | P0 | M004 | Depth Module |
| M010 | 深度质量评测 | P1 | M009 | Benchmark |
| M011 | Visual Odometry 基线 | P0 | M004,M006 | VO Module |
| M012 | VIO 融合 | P1 | M008,M011 | VIO Module |
| M013 | SLAM 建图 | P0 | M012 | SLAM Map |
| M014 | 回环检测 | P1 | M013 | Loop Closure |
| M015 | 点云生成 | P0 | M009,M013 | Point Cloud |
| M016 | 点云配准与地图融合 | P0 | M015 | Global Map |
| M017 | Mesh 重建 | P2 | M016 | Mesh |
| M018 | Occupancy Map | P1 | M016 | Voxel Map |
| M019 | 目标检测 | P0 | M004 | 2D Objects |
| M020 | 实例分割 | P1 | M019 | Instance Masks |
| M021 | 2D→3D目标关联 | P0 | M015,M019 | 3D Objects |
| M022 | 3D目标跟踪 | P0 | M021 | Object Tracks |
| M023 | 目标姿态估计 | P2 | M021 | 6DoF Objects |
| M024 | 语义点云 | P0 | M015,M021 | Semantic Point Cloud |
| M025 | 语义地图 | P1 | M024 | Semantic Map |
| M026 | Scene Graph | P0 | M021,M025 | Graph |
| M027 | 空间关系计算 | P0 | M026 | Relation Engine |
| M028 | 可见性/遮挡计算 | P1 | M026 | Visibility Model |
| M029 | 世界坐标与对象坐标 API | P0 | M013,M021 | World Model API |
| M030 | 时间状态管理 | P0 | M022,M029 | Temporal State |
| M031 | Spatial Memory | P0 | M030,M026 | Memory Store |
| M032 | 场景变化检测 | P1 | M030 | Change Detector |
| M033 | 距离/方向/路径查询 | P0 | M029,M018 | Spatial Query |
| M034 | 导航图 | P1 | M018 | Navigation Graph |
| M035 | 可供性 Affordance | P2 | M025,M026 | Affordance Layer |
| M036 | World Model 序列化 | P0 | M029,M031 | JSON Schema |
| M037 | LLM上下文适配器 | P0 | M036 | Context Adapter |
| M038 | LLM空间问答 Agent | P0 | M037 | QA Agent |
| M039 | Realtime语音交互 | P1 | M038 | Voice Agent |
| M040 | 空间推理工具集 | P0 | M033,M037 | Tool API |
| M041 | 轨迹/事件推理 | P1 | M031,M040 | Reasoning |
| M042 | 场景预测 | P2 | M031,M041 | Prediction |
| M043 | 主动视角选择 | P2 | M028,M033,M037 | Active Perception |
| M044 | 端到端延迟优化 | P1 | M038,M039 | Perf Report |
| M045 | 可靠性与异常恢复 | P0 | M003-M044 | Resilience |
| M046 | 安全边界与置信度 | P0 | M038,M040 | Safety Layer |
| M047 | 自动化评测平台 | P0 | M009-M046 | Eval Platform |
| M048 | MVP Demo | P0 | M038,M045,M046 | Demo |
| M049 | 室内空间智能 Demo | P1 | M048 | Scenario Demo |
| M050 | 户外/移动场景 Demo | P2 | M048 | Scenario Demo |

---

# 5. Foundation / 基础工程

## M001 — 项目工程骨架

**目标**：建立统一工程目录、依赖管理、配置系统、日志系统、测试框架。

**前置依赖**：无

**输入**：项目技术栈选择

**输出**：

```text
repo/
├── apps/
├── modules/
├── agents/
├── configs/
├── datasets/
├── tests/
├── scripts/
├── docs/
└── README.md
```

**Acceptance Criteria**：

- 可以一条命令完成安装
- 可以一条命令运行单元测试
- 有统一配置文件
- 有结构化日志
- CI 至少能执行 lint + unit test

---

## M002 — 设备输入方案确认

**目标**：明确 Osmo 360 当前可实际获得的数据源与限制。

**前置依赖**：M001

**任务**：

1. 确认实时视频输入路径
2. 确认分辨率、帧率、编码格式
3. 确认是否可获取 IMU / 元数据
4. 区分“官方可用路径”和“实验性路径”
5. 输出输入能力矩阵

**输出**：`docs/device_ingestion_spec.md`

**Acceptance Criteria**：所有能力都有来源、测试结果或明确的 UNKNOWN 标记。

---

## M003 — RTMP / 视频流接入

**目标**：建立稳定的实时视频入口。

**前置依赖**：M002

**输出**：`modules/ingest`

**要求**：支持连接、断线重连、时间戳、流状态监控。

**Acceptance Criteria**：连续运行 30 分钟无未处理异常；断流后可自动恢复。

---

## M004 — 视频解码与抽帧

**目标**：将视频流转换为内部标准 Frame 对象。

**输出格式建议**：

```json
{
  "frame_id": 1001,
  "timestamp": 1234567890,
  "image": "...",
  "fps": 2.0,
  "source": "osmo360"
}
```

**Acceptance Criteria**：支持可配置 FPS；时间戳单调；丢帧可统计。

---

## M005 — 数据集与回放系统

**目标**：所有算法支持离线 replay，避免每次测试都依赖真实相机。

**输出**：录制数据格式、回放器、时间同步器。

**Acceptance Criteria**：实时数据与离线回放使用同一上层 Frame API。

---

# 6. Camera / Projection / Sensor

## M006 — 相机内参标定

**目标**：得到投影模型所需内参与畸变参数。

**输出**：Calibration JSON。

**Acceptance Criteria**：提供重投影误差；误差阈值可配置；标定参数可被运行时加载。

---

## M007 — 全景投影模型

**目标**：提供 equirectangular ↔ spherical ↔ camera-ray 的统一数学模型。

**核心功能**：

- 像素 → 球面射线
- 球面射线 → 3D ray
- yaw/pitch/roll 变换
- FOV 计算

**Acceptance Criteria**：已知测试点的角度误差通过阈值；数学单元测试覆盖。

---

## M008 — IMU 数据接入

**目标**：接入角速度、加速度、时间戳，并完成坐标系定义。

**Acceptance Criteria**：IMU 时间线可与视频 Frame 对齐；坐标轴方向有明确测试依据。

---

# 7. Depth / Geometry

## M009 — 深度估计基线

**目标**：建立单目/全景图深度估计基线。

**输出**：Depth Map。

**Acceptance Criteria**：可对离线数据批量推理；输出深度图与置信度；记录推理延迟。

---

## M010 — 深度质量评测

**目标**：建立真实尺度误差评测或相对深度一致性评测。

**指标建议**：Abs Rel、RMSE、δ accuracy 或场景相关指标。

**Acceptance Criteria**：至少有一个公开/自建测试集与可重复 benchmark。

---

## M011 — Visual Odometry 基线

**目标**：根据连续 RGB / panorama 估计相机运动。

**输出**：Frame-to-frame Pose。

**Acceptance Criteria**：能在回放数据上输出连续轨迹；轨迹无 NaN；失败率可统计。

---

## M012 — VIO 融合

**目标**：融合视觉 + IMU，降低纯视觉漂移。

**输出**：6DoF Pose。

**Acceptance Criteria**：与仅视觉版本比较漂移率；改善结果有 benchmark。

---

## M013 — SLAM 建图

**目标**：从连续帧构建相机轨迹与空间地图。

**输出**：Pose Graph + Map。

**Acceptance Criteria**：支持重定位；地图可保存和恢复；轨迹可视化。

---

## M014 — 回环检测

**目标**：识别相机重新回到历史位置的情况。

**Acceptance Criteria**：在包含闭环的测试数据中能够识别回环；错误闭环率有统计。

---

## M015 — 点云生成

**目标**：将 RGB + Depth + Pose 转换到统一世界坐标系。

**点格式建议**：

```text
x, y, z, r, g, b, timestamp, confidence
```

**Acceptance Criteria**：连续帧点云可正确叠加；空间尺度与坐标单位明确。

---

## M016 — 点云配准与地图融合

**目标**：融合多帧点云并抑制漂移、重复点和明显离群点。

**Acceptance Criteria**：地图在静态场景中不会随帧数显著发散。

---

## M017 — Mesh 重建

**优先级**：P2

**目标**：从点云生成可视化/碰撞使用的三角网格。

**Acceptance Criteria**：mesh 可导出 glTF/PLY/OBJ；破洞率和三角面数可统计。

---

## M018 — Occupancy Map

**目标**：建立 free / occupied / unknown 三态空间栅格。

**输出**：Voxel / Occupancy Grid。

**Acceptance Criteria**：可查询任意位置占用状态；更新后旧信息可衰减。

---

# 8. Vision / Object Understanding

## M019 — 目标检测

**目标**：建立实时对象检测模块。

**输出**：bbox、类别、置信度、timestamp。

**Acceptance Criteria**：可在设定 FPS 下稳定运行；推理延迟可测。

---

## M020 — 实例分割

**目标**：为需要精确几何投影的对象提供 mask。

**Acceptance Criteria**：输出实例 mask ID；支持与 Frame ID 对齐。

---

## M021 — 2D→3D 目标关联

**目标**：将 2D 对象转换为世界坐标中的 3D object。

**输出建议**：

```json
{
  "object_id": "chair_001",
  "class": "chair",
  "center": [x,y,z],
  "bbox_3d": {...},
  "confidence": 0.91
}
```

**Acceptance Criteria**：静态对象在连续帧中位置稳定；坐标单位统一。

---

## M022 — 3D 目标跟踪

**目标**：为同一物体在不同时间建立稳定 ID。

**输出**：轨迹、速度、状态。

**Acceptance Criteria**：短暂遮挡后可重新关联；ID switch 有统计。

---

## M023 — 目标姿态估计

**优先级**：P2

**目标**：估计物体 6DoF orientation / pose。

---

## M024 — 语义点云

**目标**：给点云附加 object / class / instance / confidence 信息。

**Acceptance Criteria**：可按 object ID、类别过滤点云。

---

## M025 — 语义地图

**目标**：建立“房间/墙/门/桌椅/人物”等可查询语义空间。

**Acceptance Criteria**：能够进行 object-to-map 查询；地图状态可持久化。

---

# 9. Scene Graph / Spatial Relation

## M026 — Semantic Scene Graph

**目标**：建立节点和边组成的场景图。

**节点示例**：Room、Door、Table、Chair、Person。

**关系示例**：

```text
inside
on
next_to
in_front_of
behind
left_of
right_of
near
far
connected_to
```

**Acceptance Criteria**：图结构可序列化；节点具备世界坐标；边具有置信度/来源。

---

## M027 — 空间关系计算

**目标**：根据 3D 几何计算 left/right/front/behind/near/far/distance 等关系。

**重要要求**：不得让 LLM 猜测基础几何关系，应由 Geometry Engine 计算。

**Acceptance Criteria**：所有关系有明确数学定义与单元测试。

---

## M028 — 可见性与遮挡计算

**目标**：判断某对象当前是否可见、被谁遮挡、从什么方向可见。

**输出**：visibility score、occlusion ratio、source camera pose。

**Acceptance Criteria**：可解释“为什么当前看不到目标”。

---

# 10. World Model

## M029 — 世界坐标与对象坐标 API

**目标**：统一 World / Camera / Object / Body 坐标系。

**API 示例**：

```text
world_to_camera()
camera_to_world()
object_pose()
object_distance()
object_direction()
```

**Acceptance Criteria**：坐标变换双向可验证；单位统一为 SI。

---

## M030 — 时间状态管理

**目标**：维护对象随时间变化的状态。

**状态示例**：位置、速度、可见性、最后观测时间、置信度。

---

## M031 — Spatial Memory

**目标**：建立跨帧、跨位置、跨时间的空间记忆。

**必须保存**：

- Object identity
- Location
- Geometry
- Relations
- Last observed time
- Observation history
- Confidence

**Acceptance Criteria**：系统可以回答“刚才那个物体现在在哪里”。

---

## M032 — 场景变化检测

**目标**：检测新增、删除、移动、状态改变的对象。

**Acceptance Criteria**：输出 Change Event，且具备 timestamp。

---

## M033 — 距离/方向/路径查询

**目标**：提供确定性的空间查询工具。

**API 示例**：

```text
where_is(object)
distance(a,b)
direction(a,b)
nearest(object, class)
objects_inside(region)
path_to(object)
```

**Acceptance Criteria**：所有结果能够追溯到几何数据。

---

## M034 — 导航图

**优先级**：P1

**目标**：由 Occupancy Map 生成可通行空间和拓扑导航图。

**Acceptance Criteria**：支持起点→终点路径查询；碰撞检查通过。

---

## M035 — Affordance / 可供性

**优先级**：P2

**目标**：建立物体与空间的潜在操作能力，例如：

```text
chair -> can_sit
floor -> can_walk
shelf -> can_place
Door -> can_open
```

**Acceptance Criteria**：语义知识与实际几何状态分离存储。

---

## M036 — World Model 序列化

**目标**：定义 LLM 可消费的稳定 JSON Schema。

**必须包含**：

```text
scene_id
coordinate_system
camera_pose
objects
relations
regions
occupancy
memory
confidence
timestamp
```

**Acceptance Criteria**：Schema 有版本号；向后兼容策略明确。

---

# 11. LLM / Agent

## M037 — LLM 上下文适配器

**目标**：将 World Model 转换成适合 LLM 的最小必要上下文。

**原则**：

- 不发送无关点云
- 几何事实优先
- 保留置信度
- 保留时间信息
- 支持按任务检索上下文

**Acceptance Criteria**：同一问题不会无条件发送整张地图。

---

## M038 — LLM 空间问答 Agent

**目标**：实现基础空间问答。

**示例问题**：

```text
“我左边是什么？”
“桌子离我多远？”
“刚才那个人去哪了？”
“门在哪里？”
```

**Acceptance Criteria**：答案能够引用 World Model 中的证据字段。

---

## M039 — Realtime 语音交互

**优先级**：P1

**目标**：接入实时语音输入与语音输出。

**Acceptance Criteria**：语音问答链路可连续交互；超时、断线和重复输入有处理。

---

## M040 — 空间推理工具集

**目标**：不要让 LLM 自己计算精确空间数学，而是提供工具。

**工具示例**：

```text
get_object()
get_position()
get_distance()
get_direction()
get_neighbors()
get_visibility()
get_history()
find_path()
query_region()
```

**Acceptance Criteria**：LLM 可以根据任务自主调用工具；工具结果有来源与时间戳。

---

## M041 — 轨迹与事件推理

**目标**：基于 Spatial Memory 推理“发生了什么”。

**示例**：

```text
Person A
08:00 entrance
08:01 near machine
08:02 left machine
```

推理结果：

```text
Person A approached the machine and then left.
```

**Acceptance Criteria**：时间顺序正确；事件推理不得与底层观测冲突。

---

## M042 — 场景预测

**优先级**：P2

**目标**：预测移动对象短期未来位置或状态。

**输出**：预测轨迹 + uncertainty。

---

## M043 — 主动视角选择

**优先级**：P2

**目标**：让 Agent 主动决定“看哪里、什么时候看、看多久”。

**策略示例**：

```text
Global panorama
    ↓
Find uncertainty
    ↓
Select viewpoint
    ↓
Acquire high-quality observation
    ↓
Update World Model
```

**Acceptance Criteria**：相比固定多视角采样，在相同图像预算下获得更高目标确认率。

---

# 12. Reliability / Safety / Evaluation

## M044 — 端到端延迟优化

**目标**：测量并优化：

```text
Camera
→ Decode
→ Vision
→ 3D
→ World Model
→ LLM
→ Response
```

**Acceptance Criteria**：提供 P50/P95 latency；每个阶段单独计时。

---

## M045 — 可靠性与异常恢复

**目标**：所有模块支持：

- 输入断流
- 推理失败
- 时间戳异常
- 数据缺失
- GPU/CPU异常
- LLM超时

**Acceptance Criteria**：关键模块不会因为单个异常导致整个系统崩溃。

---

## M046 — 安全边界与置信度

**目标**：区分事实、推断、不确定信息。

**要求**：

```text
observed_fact
estimated_fact
inferred_fact
unknown
```

**Acceptance Criteria**：无法确定的空间关系不得输出确定性结论。

---

## M047 — 自动化评测平台

**目标**：建立统一 Benchmark。

**至少包含**：

### 几何

- Pose error
- Depth error
- Point cloud drift
- Map consistency

### 感知

- Detection AP
- Tracking ID switch
- 3D localization error

### 空间理解

- Direction accuracy
- Distance accuracy
- Relation accuracy
- Temporal reasoning accuracy

### LLM

- Answer correctness
- Groundedness
- Hallucination rate
- Tool-use success rate

### 系统

- FPS
- P50/P95 latency
- GPU memory
- CPU usage
- Network bandwidth

**Acceptance Criteria**：每次代码变更可自动跑核心 benchmark。

---

# 13. Demo / End-to-End Mission

## M048 — MVP Demo

**依赖**：M038,M045,M046

**目标**：完成最小闭环：

```text
Osmo 360
 ↓
Video
 ↓
Frame
 ↓
Object Detection
 ↓
Depth
 ↓
Camera Pose
 ↓
3D Object
 ↓
World Model
 ↓
LLM
 ↓
用户问题
```

**必须支持**：

1. “我前面有什么？”
2. “左边是什么？”
3. “桌子离我多远？”
4. “刚才的人在哪里？”

**Acceptance Criteria**：四个问题均可完成闭环并提供结构化证据。

---

## M049 — 室内空间智能 Demo

**优先级**：P1

**场景**：房间、办公室、仓库。

**目标**：实现：

- 物体识别
- 3D定位
- 房间关系
- 遮挡
- 距离
- 简单导航

**Acceptance Criteria**：固定测试场景下达到预先设定的空间问答准确率。

---

## M050 — 户外/移动场景 Demo

**优先级**：P2

**场景**：街道、园区、车辆移动。

**目标**：验证动态场景和复杂光照条件下的稳定性。

---

# 14. Agent 任务执行规范

每个 Coding Agent 接到 Mission 后，应生成以下标准交付结构：

```text
MISSION_REPORT.md
├── Goal
├── Assumptions
├── Files Changed
├── Implementation
├── Tests
├── Metrics
├── Known Issues
├── Remaining Risks
└── Next Missions
```

每个 Mission 最少需要：

1. 读取前置 Mission 的输出
2. 检查当前工程状态
3. 实现最小可工作的功能
4. 添加测试
5. 运行测试
6. 保存日志或 benchmark
7. 生成 Mission Report
8. 不修改与本 Mission 无关的公共接口

---

# 15. Agent 自动分派规则

## 15.1 可以并行执行的任务

以下 Mission 可以在基础工程完成后并行推进：

```text
M006 相机标定
M008 IMU接入
M009 深度估计
M019 目标检测
```

其后可并行推进：

```text
M017 Mesh
M018 Occupancy
M020 Instance Segmentation
M023 Pose Estimation
M034 Navigation
M035 Affordance
```

---

## 15.2 不应提前并行的任务

以下链路具有明显依赖，不建议 Agent 跳过：

```text
M013 SLAM
  ↓
M015 Point Cloud
  ↓
M021 2D→3D
  ↓
M026 Scene Graph
  ↓
M031 Spatial Memory
  ↓
M036 World Model
  ↓
M037 LLM Context
  ↓
M038 Spatial QA
```

---

# 16. 最小 MVP 路径

如果目标是最快验证“360相机是否能够形成可用于LLM的空间世界模型”，不要一次完成全部 Mission。

建议第一条 Agent Pipeline：

```text
M001
 ↓
M002
 ↓
M003
 ↓
M004
 ↓
M005
 ├── M006
 ├── M009
 └── M019
      ↓
M011
      ↓
M013
      ↓
M015
      ↓
M021
      ↓
M026
      ↓
M029
      ↓
M031
      ↓
M036
      ↓
M037
      ↓
M038
      ↓
M048
```

这条路线的最终目标不是生成“最漂亮的3D模型”，而是验证：

> **相机连续观察现实世界后，能否形成结构化、可查询、带空间和时间关系的 World Model，并让 LLM 基于这个 World Model 做 grounded reasoning。**

---

# 17. 推荐最终 Agent 编排

```text
                         Orchestrator Agent
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
   Perception Agent       Spatial Agent            LLM Agent
          │                       │                        │
     RGB / Depth              SLAM / 3D             Reasoning
     Detection                Scene Graph            Tool Use
     Tracking                 World Model            Dialogue
          │                       │                        │
          └───────────────────────┼────────────────────────┘
                                  ▼
                           Evaluation Agent
                                  │
                         Benchmark / QA / CI
```

## 推荐 Agent 角色

### Perception Agent

负责：M004、M009、M019、M020、M022。

### Spatial Agent

负责：M011-M018、M021、M024-M035。

### World Model Agent

负责：M029-M036。

### LLM Agent

负责：M037-M043。

### Systems Agent

负责：M001-M005、M044-M046。

### Evaluation Agent

负责：M047-M050。

### Orchestrator Agent

负责：依赖分析、Mission 分派、状态管理、验收门禁。

---

# 18. 最终 Definition of Done

整个项目达到最终完成状态，需要同时满足：

```text
[✓] 能实时获得相机数据
[✓] 能得到稳定时间戳
[✓] 能估计深度
[✓] 能估计相机位姿
[✓] 能建立3D地图
[✓] 能识别对象
[✓] 能给对象分配稳定ID
[✓] 能获得对象世界坐标
[✓] 能计算对象空间关系
[✓] 能维护跨时间Spatial Memory
[✓] 能形成Semantic Scene Graph
[✓] 能序列化World Model
[✓] LLM能够查询World Model
[✓] LLM回答空间问题时有几何依据
[✓] 能处理不确定性
[✓] 能进行基础时间推理
[✓] 有自动化Benchmark
[✓] 有端到端Demo
```

最终交付物应形成：

```text
Real World
   ↓
360 Camera
   ↓
Perception
   ↓
3D Geometry
   ↓
Semantic World Model
   ↓
Spatial Memory
   ↓
LLM Agent
   ↓
Spatial Intelligence
```
