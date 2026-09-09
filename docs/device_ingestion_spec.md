# Device Ingestion Spec — DJI Osmo 360

> Mission **M002** 产物。
> 本文档明确 Osmo 360 当前可实际获得的数据源与限制。所有能力条目都标注 **来源** 或 **UNKNOWN**。无设备时优先用离线数据。
>
> 最后更新: 2026-09-08
> 设备到手状态: **暂无设备**（先离线，等设备到位补实测列）

---

## 0. 数据来源与可信度

| 来源 ID | 描述 | 可信度 |
|---|---|---|
| S1 | DJI 官方规格页 https://www.dji.com/sg/mobile/360/specs | **高**（厂商公布） |
| S2 | DJI 开发者文档 (Cloud-API / PSDK / MSDK) | **高**（厂商公布） |
| S3 | DJI SDK 论坛 sdk-forum.dji.net | **中**（开发者口径） |
| S4 | 类似 360 相机公开 SDK（Insta360/Ricoh Theta）参照 | **中**（行业惯例） |
| S5 | 直接设备实测（待补） | **高**（但暂缺） |

---

## 1. 设备能力矩阵

### 1.1 录制能力（官方规格 S1）

| 能力 | 值 | 来源 | 备注 |
|---|---|---|---|
| 传感器 | 1/1.1 英寸方形 HDR CMOS | S1 | 专为 360° 设计，传感器利用率较传统 1 英寸提升 25% |
| 光圈 | f/1.9 | S1 | |
| ISO | 100–51200 | S1 | |
| 全景视频最大分辨率 | 7680×3840 (8K) | S1 | |
| 全景视频帧率 (8K) | 24/25/30/48/50 fps | S1 | |
| 全景视频帧率 (6K) | 24/25/30/48/50/60 fps | S1 | |
| 全景视频帧率 (4K) | 100 fps | S1 | 3840×1920 |
| 最大码率 | 170 Mbps | S1 | |
| 视频编码 | HEVC (H.265) | S1 | 容器: OSV / MP4 |
| 视频色彩 | 10-bit, D-Log M | S1 | |
| 全景照片最大 | 15520×7760 (120 MP) | S1 | 2:1 长宽比 |
| 防抖 | RockSteady 3.0 / HorizonSteady | S1 | **暗示有 IMU**（EIS 需陀螺仪） |
| 内置存储 | 128 GB (105 GB 可用) | S1 | exFAT |
| microSD | 至 1 TB | S1 | |
| 麦克风 | 4 个 | S1 | |
| 音频 | 48 kHz 16-bit AAC | S1 | |
| 防水 | IP68 / 10 m | S1 | |
| 续航 | 100 min @ 8K/30fps | S1 | |

### 1.2 实时视频输入路径

> **关键发现**：Osmo 360 是消费级运动相机，**DJI 目前未对消费级 Osmo 系列开放实时全景视频流 SDK**。下面四条路径的可行性均有限制。

| 路径 | 可行性 | 全景？ | 来源 | 备注 |
|---|---|---|---|---|
| **A. USB 文件读取（离线）** | ✅ 高 | ✅ 是 | S1 | 通过 USB-C 把相机当 U 盘挂载，直接读 OSV/MP4 文件。**MVP 主推路径**。 |
| **B. 单镜头直播模式（内置）** | ⚠️ 部分 | ❌ 否 | S1 | 官方规格页 "Single Lens - Livestream: 1080p (16:9): 1920×1080@30fps"。仅单镜头平面，**非全景**。无法满足 SLAM/3D 需求。 |
| **C. DJI Mimo App RTMP** | ❌ 否 | — | S3 | DJI Mimo 仅面向消费分享场景；Osmo 360 当前未见 RTMP 推流文档。MSDK 直播推流仅面向无人机 (Mavic/Air/Mini)，不支持 Osmo 系列。 |
| **D. DJI Cloud API / PSDK** | ❌ 否 | — | S2 | Cloud API Liveview 仅支持 DJI Dock/Pilot 2 体系（无人机 + 机场），不支持 Osmo 360。 |
| **E. 逆向 WiFi/USB 协议** | ⚠️ 实验性 | UNKNOWN | S4 | 类比 [insta360-go-ultra-sdk](https://github.com/Daiki-Iijima/insta360-go-ultra-sdk) 在 TCP:6666 用 protobuf；Osmo 360 是否暴露私有 TCP/UDP 端口→ **UNKNOWN，待设备实测**。 |

### 1.3 IMU / 元数据获取

| 能力 | 状态 | 来源 | 备注 |
|---|---|---|---|
| 设备是否内置 IMU | ✅ 是（隐含） | S1 | RockSteady/HorizonSteady 必须依赖陀螺仪；DJI 官方未在规格页单独列出 IMU 型号。 |
| 是否能从视频文件读 IMU | UNKNOWN | — | 待实测。需用 `ffprobe`/`exiftool` 检查 MP4/OSV 是否含 `camm` 轨 (Google Camera Motion Metadata Spec) 或 DJI 私有 metadata track。 |
| 是否能从 SDK 拿实时 IMU | ❌ 否（当前） | S2 | DJI 未公开 Osmo 360 的 IMU SDK 接口。 |
| EXIF/XMP 元数据 | UNKNOWN | — | 待实测。可能含 `Xmp.GPano.PoseHeadingDegrees` 等 360 元数据 (行业惯例)。 |
| GPS | ❌ 否 | S1 | 规格页未列 GPS。 |

### 1.4 控制能力

| 能力 | 状态 | 来源 | 备注 |
|---|---|---|---|
| 远程拍摄控制 | ❌ 无公开 SDK | S2/S3 | DJI Mimo App 提供蓝牙/WiFi 控制，但未开放给第三方 Python 程序。 |
| 远程参数设置 | ❌ 无公开 SDK | S2/S3 | 同上。 |
| 设备时间戳同步 | UNKNOWN | — | 待实测。视频文件 mtime 可用，帧级 PTP 同步 → 不太可能。 |

---

## 2. 离线数据集备选（无设备阶段）

MVP 路径在设备到位前用以下公开 360° 数据集替代真实相机输入。它们都能提供 RGB-D 全景 + 相机姿态，满足 M005 回放系统的输入要求。

| 数据集 | 规模 | 全景？ | 深度？ | 姿态？ | 语义？ | 接入 | 备注 |
|---|---|---|---|---|---|---|---|
| **RealSee3D** | 10,000 场景 / 299K 视角 | ✅ 全景 RGB | ✅ LiDAR | ✅ 全局配准 | ✅ 像素级分割 | 需签 DUA 邮件申请 | 室内住宅，HDR 拼接，深度精度最高。**MVP 主推**。 [GitHub](https://github.com/realsee-developer/RealSee3D) |
| **Matterport3D** | 90 栋建筑 / 10,800 全景 | ✅ 全景 RGB | ✅ RGB-D | ✅ 全局对齐 | ✅ 区域/对象级 | 需签协议填表 | 经典基线数据集，覆盖面广。[项目页](https://ar5iv.labs.arxiv.org/html/1709.06158) |
| **Pano3D** | 基于 M3D+GibsonV2 渲染 | ✅ 1024×512/512×256 | ✅ 深度+法向 | ✅ | 部分 | 需填表+Zenodo | 360° 深度估计专用 benchmark，[项目页](https://vcl3d.github.io/Pano3D/) |
| **ScanNet++** | ~200 室内场景 | ✅ 鱼眼全景 | ✅ LiDAR | ✅ | ✅ 语义实例 | 需签协议 | 含 fisheye 视角，[Depth Any Camera](https://github.com/yuliangguo/depth_any_camera) 已在用。 |
| **KITTI360** | 户外驾驶场景 | ✅ 鱼眼 360 | ✅ LiDAR | ✅ | ✅ 语义 | 公开 | 户外/动态场景，M050 户外 Demo 可用。 |

**MVP 第一批离线数据策略**：
1. 优先申请 **RealSee3D**（深度质量最高，对 M009 深度评测有利）
2. 同时签 **Matterport3D** 协议作为 fallback
3. KITTI360 公开下载，留作 M050 户外场景验证

---

## 3. MVP 阶段输入方案（设备到位前）

```text
[RealSee3D / Matterport3D 全景 RGB-D 文件]
    ↓ (M005 Replay System 读取)
统一 Frame 对象 (frame_id, timestamp, image, depth, pose, source="realsee3d" | "matterport3d")
    ↓ (M006+ 各 Mission 消费)
```

**Frame 字段对齐策略**（与 M004 视频解码与抽帧保持一致）：

```json
{
  "frame_id": 1001,
  "timestamp": 1234567890.0,
  "image": "<np.ndarray (H, W, 3) uint8, equirectangular panorama>",
  "depth": "<np.ndarray (H, W) float32, meters, None if unavailable>",
  "pose": "<4x4 world_to_camera SE3 matrix, None if unavailable>",
  "intrinsics": {"type": "equirect", "width": 7680, "height": 3840},
  "source": "realsee3d",
  "scene_id": "realsee3d_0001_room03_view12"
}
```

设备到位后只需把 `source` 切换为 `"osmo360_offline"`，其他字段保持不变，上层算法无需改动。

---

## 4. 设备到位后的实测 TODO

设备到手后必须补完以下条目，把 `UNKNOWN` 替换为实测值：

- [ ] **USB 文件读取**：相机挂载是否需要驱动？路径格式？OSV 文件能否用标准 ffmpeg 解码？
- [ ] **实时视频流可行性复测**：复测路径 B（单镜头直播）是否可用？是否有隐藏的全景直播模式？
- [ ] **IMU 元数据**：用 `ffprobe -show_streams -show_data <file>` 检查是否有 `camm` 或私有 track。若有，记录其采样率、坐标系定义。
- [ ] **WiFi 端口扫描**：用 `nmap` 扫描相机 WiFi IP 的常见端口（6666/80/554/8554/11009/12345），看是否暴露私有 API。
- [ ] **EXIF/XMP**：用 `exiftool -j <file>` 列出所有 360/姿态相关元数据。
- [ ] **时钟同步**：相机 RTC 漂移量；视频 timestamp 与墙钟时间关系。

实测完成后，将本文件 §1.2 / §1.3 / §1.4 的 UNKNOWN 替换为 S5 来源条目。

---

## 5. 风险与对 MVP 路径的影响

| 风险 | 对 MVP 影响 | 缓解 |
|---|---|---|
| 实时全景视频流不可用 | M003 RTMP/视频流接入无法直接面向 Osmo 360 | MVP 走离线 replay 路径；M003 改为通用 RTMP 客户端框架（可读 MediaMTX/rtmp-server），等设备或外置 360 摄像头补实测 |
| IMU 数据不可用 | M008 IMU 接入、M012 VIO 融合无法直接做 | 离线数据集自带姿态可代替 VIO；或用纯视觉 VO（M011）走 M013 SLAM；M008 改为读取数据集元数据 |
| DJI 不开放控制 API | 无法远程触发录制 | MVP 不依赖远程触发；用手动录制 + USB 拷贝的工作流 |
| 单镜头直播只有 1080p30 平面 | 无法做全景直播 | 不作为 MVP 输入路径 |

**结论**：MVP 路径在无设备/无 SDK 情况下完全可行——离线数据集提供等价的 RGB-D + pose，能完成从 M005 到 M038 的全部闭环。设备到位后切换数据源即可。
