# MISSION_REPORT — M002 设备输入方案确认

## Goal

明确 DJI Osmo 360 当前可实际获得的数据源与限制，输出 `docs/device_ingestion_spec.md`，所有能力都有来源、测试结果或明确的 UNKNOWN 标记，并给出无设备阶段的离线数据集替代方案。

## Assumptions

- 设备状态：暂无 DJI Osmo 360 实体设备，等设备到位后补实测列
- 推进策略：MVP 路径优先打通端到端闭环，不依赖实时设备
- 调研基础：DJI 官方规格页 + DJI 开发者文档 + DJI SDK 论坛 + 行业惯例类比

## Files Changed

新增:

- `docs/device_ingestion_spec.md` (M002 主产物)
- `docs/missions/M001_report.md` (从根目录迁入, 不修改内容)
- `MISSION_REPORT.md` (本文件, 改写为 M002 报告)

## Implementation

### 1. 数据来源与可信度
5 个来源 ID (S1 厂商规格 / S2 厂商 SDK 文档 / S3 论坛 / S4 行业类比 / S5 实测待补)，所有能力条目都标注来源或 UNKNOWN。

### 2. 设备能力矩阵（详见 spec §1）
- **录制能力**：8K 全景 24/25/30/48/50fps，HEVC，10-bit D-Log M，170Mbps — 来自 S1，可信度高。
- **实时视频输入路径**：5 条路径评估：
  - ✅ USB 文件读取（离线）— MVP 主推
  - ⚠️ 单镜头 1080p30 直播 — 非全景，不可用于 SLAM
  - ❌ DJI Mimo RTMP — 不支持 Osmo 360
  - ❌ Cloud API / PSDK — 仅无人机 + 机场
  - ⚠️ 逆向 WiFi 协议 — UNKNOWN，待实测
- **IMU/元数据**：设备必有 IMU（用于 RockSteady/HorizonSteady），但能否从视频文件或 SDK 读取 → UNKNOWN，待实测。
- **控制能力**：无公开远程控制 SDK。

### 3. 离线数据集（详见 spec §2）
推荐 5 个公开 360° RGB-D 数据集替代真实相机：
- RealSee3D（主推，深度精度最高）
- Matterport3D（经典基线）
- Pano3D（深度专用 benchmark）
- ScanNet++（鱼眼全景 + LiDAR）
- KITTI360（户外动态场景）

### 4. MVP 阶段输入方案（spec §3）
统一 Frame 对象字段，从离线数据集读取，设备到位后只切换 `source` 字段。

### 5. 设备到位后的实测 TODO（spec §4）
6 项实测任务清单，把所有 UNKNOWN 替换为 S5 来源条目。

## Tests

无需单元测试（M002 是调研/文档型 Mission）。验收通过文档检查进行。

## Metrics

| 指标 | 值 |
|---|---|
| spec 总字数 | ~3500 |
| 能力条目数 | 39 项 |
| 标注 UNKNOWN 条目数 | 7 项 |
| 标注 ❌ 不可用条目数 | 6 项 |
| 推荐离线数据集 | 5 个 |
| MVP 路径可达性 | ✅ 完全可行（不依赖设备） |

## Known Issues

- DJI Osmo 360 是 2025-2026 较新品，公开 SDK 文档稀少；本 spec 多处依赖 S4（行业类比）+ S1（规格页推断），可信度有限。
- "Osmo 360 是否暴露私有 TCP API" 这一项若设备实测有，会成为 M003 的潜在实时路径，但当前不作为 MVP 假设。

## Remaining Risks

- 设备到手后若发现 USB 文件读取被锁定（DJI 加密 OSV），M005 回放需先做解封装层；此项 UNKNOWN 留作 M005 风险。
- 若 IMU 元数据不可获取，M008/M012 需改为纯视觉 SLAM 路径，M011 VO 基线重要性提升。

## Next Missions

- **M003 RTMP/视频流接入**：按 spec §1.2 评估，Osmo 360 无直接实时全景流。M003 改为通用 RTMP/RTSP 客户端框架（不绑定具体相机），主要服务于未来外置 360 摄像头或 MediaMTX 中转。可参考 [aircast](https://github.com/getrismine/aircast)（DJI 直播接收，无人机）。
- **M004 视频解码与抽帧**：先支持离线 MP4/OSV 文件 → Frame；用 ffmpeg/PyAV 解 HEVC。设备到位后无缝接入。
- **M005 数据集与回放系统**：直接对接 RealSee3D/Matterport3D 文件结构，输出统一 Frame API。MVP 真正起点。
