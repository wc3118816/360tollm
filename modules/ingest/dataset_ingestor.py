"""M005 DatasetIngestor: 读离线数据集, 输出与实时 Ingestor 相同的 Frame API.

读 manifest.json + rgb/depth/pose 文件, 产出 Frame 对象.
这是 M005 AC 的核心: "实时数据与离线回放使用同一上层 Frame API".

用法:
    from modules.ingest import DatasetIngestor
    ing = DatasetIngestor("datasets/synthetic_indoor_v1")
    ing.start()
    for frame in ing.iter_frames():
        # frame.image / frame.depth / frame.pose 全部就位
        ...
    ing.stop()

支持 loop=True 循环回放, 用于长时间稳定性测试.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.config.settings import IngestCfg
from modules.ingest.base import Ingestor
from modules.ingest.types import Frame
from modules.logging import get_logger


class DatasetIngestor(Ingestor):
    """离线数据集回放 Ingestor.

    读 manifest.json 格式数据集 (见 docs/dataset_format_spec.md),
    产出 Frame 对象, 字段与 RTMPIngestor/FileIngestor 完全一致.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        cfg: IngestCfg | None = None,
        loop: bool = False,
    ) -> None:
        super().__init__(source_tag="dataset", cfg=cfg)
        self._root = Path(dataset_root).expanduser().resolve()
        self._manifest_path = self._root / "manifest.json"
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found: {self._manifest_path}")

        self._loop = loop
        self._manifest: dict[str, Any] = {}
        self._frames_meta: list[dict[str, Any]] = []
        self._cursor: int = 0  # 当前读到的帧索引
        self._intrinsics: dict[str, Any] = {}
        self._log = get_logger("modules.ingest.dataset")
        self._log.info("DatasetIngestor configured", root=str(self._root), loop=loop)

    def describe_source(self) -> dict[str, Any]:
        return {
            "type": self._source_tag,
            "dataset_root": str(self._root),
            "dataset_name": self._manifest.get("dataset_name", "unknown"),
            "dataset_type": self._manifest.get("dataset_type", "unknown"),
            "fps": self._manifest.get("fps", 0.0),
            "frame_count": self._manifest.get("frame_count", 0),
            "loop": self._loop,
        }

    # ---------------- 子类接口实现 ----------------

    def _open_stream(self) -> None:
        """加载 manifest.json + intrinsics.json."""
        try:
            self._manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ConnectionError(f"failed to read manifest: {e}") from e

        self._frames_meta = self._manifest.get("frames", [])
        if not self._frames_meta:
            raise ConnectionError(f"manifest has no frames: {self._manifest_path}")

        # 加载 intrinsics.json (可选).
        intrinsics_file = self._manifest.get("intrinsics_file")
        if intrinsics_file:
            intr_path = self._root / intrinsics_file
            if intr_path.exists():
                try:
                    self._intrinsics = json.loads(intr_path.read_text(encoding="utf-8"))
                except Exception as e:
                    self._log.warning("failed to load intrinsics", error=str(e))

        self._cursor = 0
        self._log.info(
            "dataset opened",
            name=self._manifest.get("dataset_name"),
            frames=len(self._frames_meta),
            fps=self._manifest.get("fps"),
            has_depth=self._manifest.get("has_depth"),
            has_pose=self._manifest.get("has_pose"),
        )

    def _read_next_frame(self) -> Frame:
        """按 manifest 顺序读下一帧, 加载 RGB/Depth/Pose."""
        if self._cursor >= len(self._frames_meta):
            if self._loop:
                self._log.info("dataset EOF, looping")
                self._cursor = 0
            else:
                raise StopIteration

        meta = self._frames_meta[self._cursor]
        self._cursor += 1

        # 加载 RGB.
        rgb_path = meta.get("rgb")
        image = self._load_rgb(rgb_path) if rgb_path else None

        # 加载 Depth.
        depth_path = meta.get("depth")
        depth = self._load_depth(depth_path) if depth_path else None

        # 加载 Pose.
        pose_path = meta.get("pose")
        pose = self._load_pose(pose_path) if pose_path else None

        fps = self._manifest.get("fps", 0.0)
        # DatasetIngestor 直接构造 Frame, 因为它有 depth/pose (基类 _make_frame 不带).
        return Frame(
            frame_id=self._next_frame_id(),
            timestamp=meta.get("timestamp", 0.0),
            image=image,
            fps=fps,
            depth=depth,
            pose=pose,
            source=self._source_tag,
            intrinsics=self.describe_source(),
            metadata={
                "dataset_name": self._manifest.get("dataset_name", ""),
                "dataset_type": self._manifest.get("dataset_type", ""),
                "frame_index": self._cursor - 1,
                "depth_unit": self._manifest.get("depth_unit", "mm"),
            },
        )

    def _close_stream(self) -> None:
        # 无资源需释放 (manifest 已在内存).
        self._cursor = 0

    # ---------------- 文件加载辅助 ----------------

    def _load_rgb(self, rel_path: str | None) -> Any:
        """加载 RGB 图像为 (H, W, 3) uint8 ndarray."""
        if not rel_path:
            return None
        abs_path = self._root / rel_path
        if not abs_path.exists():
            self._record_drop(f"rgb_missing:{rel_path}")
            return None
        try:
            import numpy as np
            from PIL import Image  # type: ignore

            img = Image.open(abs_path).convert("RGB")
            return np.array(img)
        except Exception as e:
            self._record_drop(f"rgb_load_error:{e}")
            self._log.debug("rgb load failed", path=rel_path, error=str(e))
            return None

    def _load_depth(self, rel_path: str | None) -> Any:
        """加载深度图为 (H, W) float32 米单位 ndarray."""
        if not rel_path:
            return None
        abs_path = self._root / rel_path
        if not abs_path.exists():
            self._record_drop(f"depth_missing:{rel_path}")
            return None
        try:
            import numpy as np
            from PIL import Image  # type: ignore

            img = Image.open(abs_path)
            arr = np.array(img, dtype=np.float32)
            # mm → m 转换 (若 manifest 声明 mm).
            unit = self._manifest.get("depth_unit", "mm")
            if unit == "mm":
                arr = arr / 1000.0
            return arr
        except Exception as e:
            self._record_drop(f"depth_load_error:{e}")
            self._log.debug("depth load failed", path=rel_path, error=str(e))
            return None

    def _load_pose(self, rel_path: str | None) -> Any:
        """加载 4x4 位姿矩阵为 float64 ndarray."""
        if not rel_path:
            return None
        abs_path = self._root / rel_path
        if not abs_path.exists():
            self._record_drop(f"pose_missing:{rel_path}")
            return None
        try:
            import numpy as np

            data = json.loads(abs_path.read_text(encoding="utf-8"))
            w2c = data.get("world_to_camera")
            if w2c is None:
                return None
            return np.array(w2c, dtype=np.float64)
        except Exception as e:
            self._record_drop(f"pose_load_error:{e}")
            self._log.debug("pose load failed", path=rel_path, error=str(e))
            return None

    # ---------------- 公开属性 ----------------

    @property
    def intrinsics(self) -> dict[str, Any]:
        """数据集全局内参 (从 intrinsics.json)."""
        return self._intrinsics

    @property
    def frame_count(self) -> int:
        """数据集总帧数."""
        return len(self._frames_meta)
