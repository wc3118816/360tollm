# 离线数据集格式规范 (M005)

## 目标

定义统一的离线数据集目录结构 + manifest 格式，让 DatasetIngestor 能读取并产出与实时 Ingestor（RTMPIngestor / FileIngestor）**完全相同**的 Frame API。这是 M005 AC 的核心："实时数据与离线回放使用同一上层 Frame API"。

## 目录结构

```
datasets/<dataset_name>/
├── manifest.json          # 数据集元数据 + 帧索引
├── rgb/                   # RGB 图像序列 (PNG/JPG)
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/                 # 深度图 (可选, 16-bit PNG, 毫米单位)
│   ├── 000000.png
│   └── ...
├── pose/                  # 位姿文件 (可选, 4x4 numpy .npy 或 JSON)
│   ├── 000000.json        # {"world_to_camera": [[...]]} 4x4 矩阵
│   └── ...
├── intrinsics.json        # 相机内参 (全局, 可被帧级覆盖)
└── metadata.json          # 数据集来源/采集信息 (可选)
```

## manifest.json 格式

```json
{
  "dataset_name": "synthetic_indoor_v1",
  "dataset_type": "synthetic|realsee3d|matterport3d|osmo360_offline",
  "fps": 30.0,
  "frame_count": 300,
  "width": 640,
  "height": 480,
  "has_depth": true,
  "has_pose": true,
  "depth_unit": "mm",
  "intrinsics_file": "intrinsics.json",
  "frames": [
    {
      "frame_id": 1,
      "timestamp": 0.0,
      "rgb": "rgb/000000.png",
      "depth": "depth/000000.png",
      "pose": "pose/000000.json"
    },
    {
      "frame_id": 2,
      "timestamp": 0.033,
      "rgb": "rgb/000001.png",
      "depth": "depth/000001.png",
      "pose": "pose/000001.json"
    }
  ]
}
```

- `timestamp`：浮点秒，单调递增
- `rgb`/`depth`/`pose`：相对数据集根目录的路径；`null` 表示该帧缺该流
- `depth_unit`：`mm`（毫米，PNG 16-bit）/ `m`（米，float32 .npy）

## intrinsics.json 格式

```json
{
  "type": "perspective|equirect",
  "fx": 525.0,
  "fy": 525.0,
  "cx": 319.5,
  "height": 480,
  "width": 640,
  "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

全景图用 `equirect` 类型：
```json
{
  "type": "equirect",
  "width": 7680,
  "height": 3840
}
```

## pose 文件格式

每帧位姿存为 JSON：
```json
{
  "world_to_camera": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

4x4 SE3 矩阵，`world_to_camera` 表示世界坐标到相机坐标的变换。

## 设计原则

1. **零依赖**：只用 PNG/JSON/npy 等通用格式，不绑定特定数据集 SDK
2. **可扩展**：`frames[].rgb/depth/pose` 可为 null，支持单流/双流/三流数据集
3. **与 Frame API 对齐**：manifest 的字段直接映射到 `Frame` 对象
4. **适配真实数据集**：RealSee3D/Matterport3D 的转换脚本输出此格式即可被 DatasetIngestor 消费

## 与实时 Ingestor 的一致性

| Frame 字段 | 实时 Ingestor 来源 | DatasetIngestor 来源 |
|---|---|---|
| frame_id | Ingestor 自增 | manifest frames[].frame_id |
| timestamp | decode PTS / wall clock | manifest frames[].timestamp |
| image | av_frame.to_ndarray | PIL.Image.open → ndarray |
| fps | cfg.target_fps / stream.average_rate | manifest.fps |
| depth | None (M003 阶段) | depth PNG / npy (如有) |
| pose | None (M003 阶段) | pose JSON (如有) |
| source | "rtmp"/"file" | manifest.dataset_type |
| intrinsics | describe_source() | intrinsics.json |
| metadata | codec/pts/width/height | 帧索引 + 数据集名 |
