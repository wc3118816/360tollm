# MISSION_REPORT — M004 视频解码与抽帧

## Goal

将视频流转换为内部标准 Frame 对象。AC：支持可配置 FPS；时间戳单调；丢帧可统计。

## Assumptions

- M003 已建立 Ingestor ABC + RTMPIngestor + FileIngestor + Frame 对象 + 抽帧节流
- M004 是 M003 的增量：补全 fps 字段、丢帧统计、帧 pipeline 中间件
- 时间戳单调性在 M003 已覆盖测试，M004 不重复

## Files Changed

新增：
- `modules/ingest/pipeline.py`（FramePipeline + PipelineStats + DedupFrames + KeyframeSelector）
- `tests/unit/test_ingest_pipeline.py`（14 单测）

修改：
- `modules/ingest/types.py`：Frame 加 `fps: float = 0.0` 字段
- `modules/ingest/base.py`：`_make_frame` 加 fps 参数（默认从 cfg.target_fps）；新增 `_record_drop()` 辅助方法
- `modules/ingest/file_ingestor.py`：抽帧节流跳帧时调 `_record_drop`；解码失败/dts=None 也记录；`_make_frame` 传入源流实际 fps（从 `stream.average_rate`）
- `modules/ingest/rtmp_ingestor.py`：同样补 `_record_drop` 调用
- `modules/ingest/__init__.py`：导出 FramePipeline / PipelineStats / DedupFrames / KeyframeSelector
- `tests/unit/test_ingest_types.py`：加 fps 字段测试 + frames_dropped 累计测试
- `tests/integration/test_ingest_e2e.py`：加 fps 字段集成测试 + throttle 丢帧统计测试
- `README.md`：状态表更新 M004 = DONE

## Implementation

### 1. Frame 加 fps 字段（types.py）
```python
@dataclass(slots=True)
class Frame:
    frame_id: int
    timestamp: float
    image: Any = None
    fps: float = 0.0  # 新增: 该帧对应的产出帧率 (Hz)
    ...
```
- FileIngestor 从 `stream.average_rate` 提取源流实际 fps（30fps）
- throttle=False 时 fps = 源流 fps；throttle=True 时 fps = target_fps（产出节奏）
- RTMPIngestor fps = target_fps（实时流无明确源帧率）

### 2. 丢帧统计（base.py + file_ingestor.py + rtmp_ingestor.py）
```python
def _record_drop(self, reason: str = "") -> None:
    self._stats.frames_dropped += 1
    if reason:
        self._log.debug("frame dropped", reason=reason, ...)
```
- **throttle_skip**：抽帧节流跳过帧时记录
- **dts_is_none**：packet.dts is None（时序缺失）时记录
- **decode_error**：解码异常时记录
- `IngestStats.frames_dropped` 累计所有丢帧原因

### 3. FramePipeline 中间件（pipeline.py）
```python
pipe = FramePipeline(ingestor, [DedupFrames(threshold=1.0), KeyframeSelector(interval_sec=2.0)])
pipe.start()
for frame in pipe.iter_frames(max_frames=100):
    ...
```
- **PipelineStats**：frames_in / frames_out / frames_filtered / drop_rate
- **中间件签名**：`(Frame, ctx) -> Frame | None`（None = 丢弃）
- **DedupFrames**：基于图像灰度均值差的简单去重（计算成本极低）
- **KeyframeSelector**：按时间间隔选关键帧（降低后续模块计算负载）
- **异常处理**：中间件抛异常时该帧被丢弃，pipeline 不崩
- **短-circuit**：中间件返回 None 后不再调用后续中间件
- **叠加 stats**：`pipe.ingestor_stats` 访问底层 Ingestor 的 frames_emitted/frames_dropped

## Tests

```bash
uv run pytest                        # 68 测试
uv run pytest -m unit                # 55 单测
uv run pytest -m integration         # 13 集成测试
uv run ruff check . && uv run ruff format --check .
```

### 测试覆盖
- **types 单测 (11)**：Frame fps 字段默认 0 + 显式构造 + frames_dropped 累计
- **pipeline 单测 (14)**：PipelineStats 初始化/drop_rate、passthrough、max_frames 限制、DedupFrames 首帧保留/相同帧丢弃/不同帧保留、KeyframeSelector 首帧/间隔过滤、多中间件串联、异常丢弃帧不崩、return None 短-circuit、ingestor_stats 叠加
- **集成测试 (13)**：FileIngestor frame.fps > 0 且 ~30fps、throttle 模式 frames_dropped > 0、throttle=False 时 frames_dropped <= 5

## Metrics

| 指标 | 值 |
|---|---|
| 总测试数 | 68 (55 unit + 13 integration) |
| 通过率 | 68/68 (100%) |
| 总耗时 | 3.80s |
| Lint | All checks passed |
| Format | 33 files already formatted |
| 代码行数 | ~250 (pipeline.py) + ~100 (测试) + ~50 (其他修改) |

## AC 对照

| AC | 实测 |
|---|---|
| 支持可配置 FPS | ✅ IngestCfg.target_fps 配置；Frame.fps 字段暴露给下游；throttle=True 时按 target_fps 节流 |
| 时间戳单调 | ✅ M003 已覆盖（test_timestamp_non_decreasing + test_loop_stability_no_unhandled_exception） |
| 丢帧可统计 | ✅ IngestStats.frames_dropped 累计 throttle_skip/dts_is_none/decode_error；PipelineStats.frames_filtered 累计中间件丢弃 |

## Remaining Risks

- DedupFrames 用灰度均值差，对复杂场景去重能力有限。M009+ 阶段可升级为感知哈希（pHash）。
- KeyframeSelector 按墙钟时间戳，对 PTS 跳变敏感。M005 replay 阶段改用 decode PTS。
- 8K 全景 (7680x3840) 的 DedupFrames 计算成本：np.mean on 7680*3840*3 ≈ 88M elements，约 50ms/帧。可优化为下采样后计算。

## Next Missions

- **M005 数据集与回放系统**：DatasetIngestor 子类读 RealSee3D/Matterport3D 文件结构，输出统一 Frame API。MVP 真正起点。FramePipeline 已为 M005 铺路（replay 可挂载 DedupFrames/KeyframeSelector 降负载）。
- **M006 相机内参标定**：Frame.intrinsics 字段已就位，M006 填入标定结果。
- **M009 深度估计基线**：FramePipeline + KeyframeSelector 已降低深度模块的输入帧率负载。
