# MISSION_REPORT — M005 数据集与回放系统

## Goal

所有算法支持离线 replay，避免每次测试都依赖真实相机。AC："实时数据与离线回放使用同一上层 Frame API"。

## Assumptions

- M004 已建立 Frame API（含 fps/depth/pose 字段）+ FramePipeline
- 无设备阶段用合成数据集替代真实数据集（RealSee3D/Matterport3D 需数 GB 下载）
- 数据集格式自研，但兼容真实数据集转换（manifest.json 通用格式）

## Files Changed

新增：
- `docs/dataset_format_spec.md`（离线数据集格式规范）
- `modules/ingest/dataset_ingestor.py`（DatasetIngestor）
- `modules/ingest/timesync.py`（TimeSync + SyncStats）
- `scripts/make_synthetic_dataset.py`（合成数据集生成脚本）
- `tests/unit/test_dataset_ingestor.py`（13 单测）
- `tests/unit/test_timesync.py`（13 单测）
- `tests/integration/test_dataset_replay_e2e.py`（9 集成测试）
- `datasets/synthetic_indoor_v1/`（30 帧合成数据集：RGB + Depth + Pose + intrinsics）

修改：
- `modules/ingest/__init__.py`：导出 DatasetIngestor / TimeSync / SyncStats
- `pyproject.toml`：dev 组加 pillow + numpy（生成脚本依赖）
- `README.md`：状态表更新 M005 = DONE

## Implementation

### 1. 数据集格式规范（docs/dataset_format_spec.md）

```
datasets/<name>/
├── manifest.json          # 元数据 + 帧索引
├── rgb/000000.png         # RGB 图像序列
├── depth/000000.png       # 16-bit 深度 (mm)
├── pose/000000.json       # 4x4 位姿矩阵
└── intrinsics.json        # 相机内参
```

manifest.json 字段直接映射 Frame 对象：frame_id / timestamp / image / depth / pose / fps / source / intrinsics。

### 2. DatasetIngestor（dataset_ingestor.py）
- 读 manifest.json + intrinsics.json
- 按 frames[] 顺序加载 RGB（PNG → ndarray）/ Depth（PNG → float32 米）/ Pose（JSON → 4x4 矩阵）
- depth_unit=mm 时自动转米（÷1000）
- 文件缺失时 `_record_drop` + 字段置 None（不崩）
- loop=True 循环回放
- describe_source 暴露 dataset_name / dataset_type / fps / frame_count
- **AC 核心**：产出 Frame 与 FileIngestor/RTMPIngestor 字段集完全一致

### 3. TimeSync 时间同步器（timesync.py）
- 多流时间对齐：主流（RGB）+ 从流（Depth/Pose）按最近邻匹配
- 二分查找最近邻帧，偏差 > max_delta_sec 时字段置 None
- SyncStats 统计 align_count / misalign_count / max_delta_sec
- 应用场景：未来多源实时流（RGB from RTMP + Depth from USB + Pose from SLAM）
- DatasetIngestor 读的 manifest 格式已预对齐，通常不需 TimeSync

### 4. 合成数据集（scripts/make_synthetic_dataset.py）
- 30 帧 @10fps，320×240
- RGB：渐变色块，帧间偏移（模拟运动视差）
- Depth：平面 2000mm（2m）
- Pose：沿 X 轴平移 0.1m/帧
- intrinsics：针孔模型 fx=fy=320

## Tests

```bash
uv run pytest                        # 105 测试
uv run pytest -m unit                # 83 单测
uv run pytest -m integration         # 22 集成测试
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖
- **dataset_ingestor 单测 (13)**：初始化（manifest 缺失/空/正常）、describe_source、按序读帧、RGB/Depth/Pose 加载、depth mm→m、source 字段、fps 字段、EOF/loop、缺失流处理、文件缺失 _record_drop
- **timesync 单测 (13)**：最近邻 depth/pose 对齐、双流同时、偏差超阈值、misalign 累计、无候选、空候选、Frame 继承 image/source/fps/timestamp、metadata.synced、统计字段
- **集成测试 (9)**：完整读 30 帧、RGB+Depth+Pose 全流、depth 米单位、pose 平移递增、timestamp 单调、loop 跨 EOF、**AC: Frame 字段集与 FileIngestor 一致**、pipeline 消费 dataset、source 区分来源

## Metrics

| 指标 | 值 |
|---|---|
| 总测试数 | 105 (83 unit + 22 integration) |
| 通过率 | 105/105 (100%) |
| 总耗时 | 37.41s |
| Lint | All checks passed |
| Format | 41 files already formatted |
| 代码行数 | ~250 (dataset_ingestor) + ~150 (timesync) + ~100 (脚本) + ~400 (测试) |
| 合成数据集 | 30 帧, ~2 MB |

## AC 对照

| AC | 实测 |
|---|---|
| 实时数据与离线回放使用同一上层 Frame API | ✅ `test_frame_fields_identical`：DatasetIngestor 与 FileIngestor 产出的 Frame 字段集完全一致（dataclass fields 相同）；`test_pipeline_consumes_both_sources`：同一 FramePipeline + DedupFrames/KeyframeSelector 对两类源都生效；`test_source_tag_distinguishes_origin`：source 字段区分来源 |

## Remaining Risks

- 合成数据集场景简单（平面深度、线性平移）。真实数据集（RealSee3D）转换脚本待 M009 阶段实现。
- TimeSync 用最近邻，未做插值。高频场景可能需线性插值（M012 VIO 阶段评估）。
- 大数据集（10000+ 帧）的 manifest.json 加载内存：当前全量载入，未来可改流式读取。

## Next Missions

- **M006 相机内参标定**：Frame.intrinsics 字段已就位，M006 填入标定结果（重投影误差阈值可配置）。
- **M009 深度估计基线**：DatasetIngestor 已能产出带 depth 的 Frame，M009 可直接用合成数据集验证深度模型（Metric3D）。
- **M011 Visual Odometry**：DatasetIngestor 产出的 pose 可作为 VO 模块的 ground truth 评估基准。
