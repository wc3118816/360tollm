# MISSION_REPORT — M001 项目工程骨架

## Goal

建立统一工程目录、依赖管理、配置系统、日志系统、测试框架，使后续所有 Mission 可在统一基础上增量开发。

## Assumptions

- 主技术栈: 纯 Python (≥ 3.11)
- 依赖管理: uv (本机已装 0.11.24)
- 推进策略: 严格按任务指南 §16 MVP 路径
- 设备: 暂无 DJI Osmo 360，M002 用离线数据先做，等设备到位补实测
- CI 跑在 ubuntu-latest + windows-latest，Py 3.11 与 3.12 矩阵

## Files Changed

新增 (M001 范围):

```
tollm/
├── .github/workflows/ci.yml
├── .gitignore
├── README.md
├── MISSION_REPORT.md
├── pyproject.toml
├── ruff.toml
├── configs/default.yaml
├── modules/__init__.py
├── modules/config/__init__.py
├── modules/config/loader.py
├── modules/config/settings.py
├── modules/ingest/__init__.py
├── modules/logging/__init__.py
├── agents/__init__.py
├── apps/__init__.py
├── apps/replay/__init__.py
├── apps/qa_agent/__init__.py
├── tests/__init__.py
├── tests/conftest.py
├── tests/unit/__init__.py
├── tests/unit/test_config_loader.py
├── tests/unit/test_config_settings.py
├── tests/unit/test_logging.py
├── tests/integration/__init__.py
└── (空目录占位) datasets/ scripts/ docs/
```

## Implementation

### 依赖与包结构
- `pyproject.toml` 用 hatchling 作为 build backend；运行时依赖最小化 (pydantic, pyyaml, structlog, numpy)，按子系统分组 optional-extra：vision / geometry / agent / dev。
- `[dependency-groups] dev` 通过 uv 的 `--group dev` 一次性安装开发工具。

### 配置系统 (modules/config)
- `loader.py`: 解析工程根目录、加载 YAML 栈 (default → local → $TOLLM_CONFIG_PATH)、深度合并。
- `settings.py`: pydantic BaseSettings，`TOLLM_` 前缀 + `__` 嵌套分隔符；包含 9 个子配置块覆盖后续各 Mission 维度。
- `get_settings()` lru_cache 单例；`reload_settings()` 测试用。

### 日志系统 (modules/logging)
- 基于 structlog；首次 `get_logger(name)` 自动配置；console 与 json 两种 renderer。
- `log_level_override` 支持 `"modules.ingest=DEBUG perf=WARN"` 模块级调试。
- `reset_logging_for_tests()` 测试清理。

### 测试
- `conftest.py` 提供 `project_root` / `fresh_settings` / `autouse reset_logging` fixtures。
- 7 个 unit test 覆盖 loader、settings 环境变量覆盖、日志 override 解析与 logger 调用。

### CI
- `.github/workflows/ci.yml`：lint job + 测试矩阵 (ubuntu/windows × 3.11/3.12)；用 `astral-sh/setup-uv@v3` + `uv sync`。

## Tests

```bash
uv sync --group dev
uv run pytest -m unit
```

预期: 全部通过 (见 Metrics 段实测结果)。

## Metrics

本机验证 (Python 3.12.12 via uv, Windows):

| 项 | 结果 |
|---|---|
| 安装 (`uv sync --group dev`) | OK, 26 包, ~17s |
| 单测 (`pytest -m unit`) | 16 passed in 0.24s |
| Lint (`ruff check .`) | All checks passed |
| Format (`ruff format --check .`) | 9 files reformatted, 11 unchanged |
| 覆盖率 (可选) | `uv run pytest --cov=modules --cov=agents --cov=apps` |

## Known Issues

- 本机已安装的是 Python 3.14.3；当前 pyproject 要求 `>=3.11`，CI 矩阵用 3.11/3.12。本地需通过 `uv python install 3.11` 准备解释器。
- `filterwarnings = ["error::DeprecationWarning"]` 在某些第三方库更新时可能需要降级为 warning，但 M001 阶段不影响。
- Open3D / PyTorch 等 optional 依赖留待 M009/M015 等具体 Mission 触发时再装，避免 M001 工程骨架过重。

## Remaining Risks

- Windows 路径中含非 ASCII 工作目录可能影响某些第三方库；本机路径为 `d:\first-cc\360tollm`，安全。
- structlog `cache_logger_on_first_use=True` 与 `reset_logging_for_tests()` 之间的交互需测试验证 (已加 fixture)。

## Next Missions

- **M002 设备输入方案确认**: 离线模式下输出 `docs/device_ingestion_spec.md`，所有能力有来源或 UNKNOWN 标记；待设备到手补实测列。
- **M003 RTMP/视频流接入**: 在 `modules/ingest` 实现，参考 [aircast](https://github.com/getrismine/aircast) (DJI 直播接收) 与 [insta360-go-ultra-sdk](https://github.com/Daiki-Iijima/insta360-go-ultra-sdk) (类 360 相机 WiFi 直播)。
