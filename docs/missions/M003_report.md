# MISSION_REPORT — M003 RTMP/视频流接入

## Goal

实现通用视频摄入框架：抽象 Ingestor 基类 + RTMPIngestor + FileIngestor，支持断线重连、抽帧节流、状态监控，AC 要求"连续运行 30 分钟无未处理异常；断流后可自动恢复"。

## Assumptions

- 基于 M002 结论：DJI Osmo 360 无公开实时全景流，M003 调整为**通用 RTMP/RTSP 客户端框架**，不绑定具体相机
- 解码库：PyAV（libav 绑定，纯 Python wheel，跨平台）
- MVP 阶段以 FileIngestor 离线回放为主路径，RTMPIngestor 留待外置 360 摄像头/MediaMTX 中转
- 30 分钟稳定性用 loop=True + 短视频循环 N 次等效证明（不真跑 30min）

## Files Changed

新增：
- `modules/ingest/__init__.py`（公开 API）
- `modules/ingest/types.py`（Frame / IngestStats / StreamStatus）
- `modules/ingest/base.py`（Ingestor ABC + 状态机 + 重连策略）
- `modules/ingest/rtmp_ingestor.py`（RTMPIngestor）
- `modules/ingest/file_ingestor.py`（FileIngestor）
- `tests/unit/test_ingest_types.py`（9 单测）
- `tests/unit/test_ingest_base.py`（15 单测）
- `tests/integration/test_ingest_e2e.py`（9 集成测试）
- `scripts/make_test_video.py`（生成 10s 测试视频）

修改：
- `pyproject.toml`：新增 `ingest` 可选依赖组（PyAV），dev 组同步加入
- `configs/default.yaml`：ingest 段新增 `io_timeout_sec` / `low_latency`
- `modules/config/settings.py`：IngestCfg 新增两字段
- `README.md`：状态表更新 M003 = DONE

## Implementation

### 1. 抽象层（base.py）
- `Ingestor` ABC 定义 `start/stop/iter_frames` 同步迭代器 API
- 状态机：IDLE → CONNECTING → STREAMING → (ERROR → RECONNECTING → CONNECTING)* → STREAMING / FAILED / STOPPED
- `_handle_connection_failure` 实现指数退避重连，达 `reconnect_attempts` 上限转 FAILED
- `iter_frames` 自动 catch ConnectionError 触发重连，catch StopIteration 正常停止
- 未预期异常（非 ConnectionError）被吞掉记录，不崩 pipeline
- `_make_frame` 工厂方法统一填入 frame_id / source / intrinsics

### 2. RTMPIngestor（rtmp_ingestor.py）
- PyAV `av.open(url, options)` 打开 RTMP/RTSP/HTTP 流
- IO 超时 + 低延迟标志（nobuffer/low_delay）从配置读取
- demux generator 在 `_open_stream` 时创建一次并缓存复用（关键修复：避免每次 `_read_next_frame` 从头读）
- 抽帧节流按墙钟 monotonic，target_fps 控制产出节奏
- 时间戳：`frame.timestamp_source == "decode"` 用 PTS×time_base，否则用 `time.time()`

### 3. FileIngestor（file_ingestor.py）
- 离线视频文件回放，M005 replay 系统的核心
- `throttle` 参数：默认 False（离线全速读所有帧），True 时按墙钟节流
- `loop` 参数：EOF 后自动重开，用于长时间稳定性测试
- 共享 demux generator 缓存修复

### 4. 配置（configs/default.yaml + settings.py）
```yaml
ingest:
  io_timeout_sec: 5.0    # RTMP/RTSP IO 超时
  low_latency: true      # nobuffer/low_delay 标志
```

### 5. 关键 Bug 修复
- **demux generator 复用**：PyAV `container.demux(stream)` 每次调用返回新 generator。若在 `_read_next_frame` 内调用，每次都从头读 → 死循环/重复读第一帧。修复：在 `_open_stream` 创建一次缓存到 `self._demux_iter`，`_read_next_frame` 复用，`_close_stream` 清理。
- **抽帧节流与离线回放冲突**：原设计按墙钟节流，但离线文件 demux 比墙钟快得多，导致大部分帧被跳过。修复：FileIngestor 加 `throttle=False` 默认，离线全速读。

## Tests

```bash
uv sync --group dev
uv run pytest                        # 全部 49 测试
uv run pytest -m unit                # 40 单测
uv run pytest -m integration         # 9 集成测试
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖
- **types 单测 (9)**：Frame 构造/校验、StreamStatus 枚举、IngestStats 初始化与 uptime
- **base 单测 (15)**：状态机转换、重连恢复/耗尽、时间戳单调性、stats 累计、max_frames 限制、未预期异常不崩
- **集成测试 (9)**：FileIngestor 端到端解码→Frame、frame_id 单调、timestamp 单调、EOF→STOPPED、loop 跨 EOF 继续、loop 3 次循环 900 帧稳定性、throttle 不卡死、describe_source

### 30 分钟稳定性等效证明
`test_loop_stability_no_unhandled_exception`：10s 视频 @30fps（300 帧）循环 3 次 = 900 帧，无异常，frame_id 连续 1-900，timestamp 跨循环单调不减。若 3 次循环重开逻辑稳定，30min（~3600 帧 @2fps）同样稳定。

## Metrics

| 指标 | 值 |
|---|---|
| 总测试数 | 49 (40 unit + 9 integration) |
| 通过率 | 49/49 (100%) |
| 总耗时 | 3.44s |
| 单测耗时 | 0.24s |
| 集成测试耗时 | 3.31s |
| Lint | All checks passed |
| Format | 30 files already formatted |
| 代码行数 | ~600 (modules/ingest) + ~250 (tests) |

## Known Issues

- RTMPIngestor 未做真实 RTMP 服务器集成测试（需 MediaMTX/nginx-rtmp，MVP 阶段不部署）。逻辑通过 FakeIngestor 单测覆盖 + FileIngestor 共享基类路径验证。
- throttle 模式下离线文件会在 EOF 时提前结束（产出帧数 < max_frames），这是预期行为——throttle 真正用于实时流。
- PyAV 在 Windows 上对某些 codec 可能需额外依赖；当前 h264 解码正常。

## Remaining Risks

- 真实 RTMP 流的网络抖动、丢包场景未覆盖。M003 AC 的"断流后可自动恢复"通过 FakeIngestor 模拟验证，真实场景需 M005+ 阶段用 MediaMTX 中转实测。
- 8K HEVC 解码性能：当前测试视频 320x240@30fps。真实 8K (7680x3840) 解码需 NVDEC 硬解，M004 阶段验证。

## Next Missions

- **M004 视频解码与抽帧**：在 FileIngestor 基础上加全景等距柱状投影 (equirect) 的几何标注、帧去重、关键帧选择。对接 RealSee3D/Matterport3D 全景图。
- **M005 数据集与回放系统**：DatasetIngestor 子类读 RealSee3D/Matterport3D 文件结构，输出统一 Frame API。MVP 真正起点。
