"""M015 点云生成: PointCloud 数据结构.

承载:
- points: float32 (N, 3) 3D 点坐标 (世界坐标系).
- colors: uint8 (N, 3) RGB 颜色 (可选, 来自原图像素).
- normals: float32 (N, 3) 法向量 (可选, 后续 M016 计算).
- metadata: source frame_id / pose / depth backend / n_points 等.

序列化:
- save_ply() / load_ply()  通用 PLY 格式 (ASCII + binary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_NDIM_2D = 2
_NDIM_3D = 3
_CHANNELS_RGB = 3
_NDIM_1D = 1


@dataclass(slots=True)
class PointCloud:
    """3D 点云.

    Attributes:
        points: float32 (N, 3) 世界坐标系 3D 点.
        colors: uint8 (N, 3) RGB 颜色, 可选 (空数组表示无颜色).
        normals: float32 (N, 3) 法向量, 可选.
        metadata: 扩展字段 (source_frame_ids / depth_backend / pose_backend 等).
    """

    points: np.ndarray
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, _CHANNELS_RGB), dtype=np.uint8))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, _NDIM_3D), dtype=np.float32))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != _NDIM_2D or self.points.shape[1] != _NDIM_3D:
            raise ValueError(f"points must be (N,3), got {self.points.shape}")
        self.colors = np.asarray(self.colors, dtype=np.uint8)
        if self.colors.size == 0:
            self.colors = self.colors.reshape(0, _CHANNELS_RGB)
        if self.colors.ndim != _NDIM_2D or self.colors.shape[1] != _CHANNELS_RGB:
            raise ValueError(f"colors must be (N,3), got {self.colors.shape}")
        self.normals = np.asarray(self.normals, dtype=np.float32)
        if self.normals.size == 0:
            self.normals = self.normals.reshape(0, _NDIM_3D)
        if self.normals.ndim != _NDIM_2D or self.normals.shape[1] != _NDIM_3D:
            raise ValueError(f"normals must be (N,3), got {self.normals.shape}")
        # 颜色/法向量长度与 points 一致 (非空时).
        if self.colors.shape[0] not in (0, self.points.shape[0]):
            raise ValueError(
                f"colors length {self.colors.shape[0]} != points length {self.points.shape[0]}"
            )
        if self.normals.shape[0] not in (0, self.points.shape[0]):
            raise ValueError(
                f"normals length {self.normals.shape[0]} != points length {self.points.shape[0]}"
            )

    @property
    def n_points(self) -> int:
        """点数."""
        return self.points.shape[0]

    @property
    def has_colors(self) -> bool:
        return self.colors.shape[0] > 0

    @property
    def has_normals(self) -> bool:
        return self.normals.shape[0] > 0

    @property
    def has_nan(self) -> bool:
        """检查是否含 NaN (AC: 点云无 NaN)."""
        return not np.isfinite(self.points).all()

    def merge(self, other: PointCloud) -> None:
        """合并另一个点云 (in-place)."""
        self.points = np.vstack([self.points, other.points])
        if self.has_colors and other.has_colors:
            self.colors = np.vstack([self.colors, other.colors])
        elif other.has_colors:
            # 当前无颜色, other 有 → 用 other 的 (当前点补黑).
            self.colors = np.vstack(
                [
                    np.zeros((self.n_points - other.n_points, _CHANNELS_RGB), dtype=np.uint8),
                    other.colors,
                ]
            )
        if self.has_normals and other.has_normals:
            self.normals = np.vstack([self.normals, other.normals])
        # metadata 合并.
        for k, v in other.metadata.items():
            if k in self.metadata:
                if isinstance(self.metadata[k], list) and isinstance(v, list):
                    self.metadata[k].extend(v)
                else:
                    self.metadata[k] = [self.metadata[k], v]
            else:
                self.metadata[k] = v

    def filter_valid(self) -> None:
        """移除 NaN/Inf 点 (in-place)."""
        mask = np.isfinite(self.points).all(axis=1)
        self.points = self.points[mask]
        if self.has_colors:
            self.colors = self.colors[mask]
        if self.has_normals:
            self.normals = self.normals[mask]

    def summary(self) -> dict[str, Any]:
        """统计摘要."""
        valid = np.isfinite(self.points).all(axis=1)
        n_valid = int(valid.sum())
        return {
            "n_points": self.n_points,
            "n_valid": n_valid,
            "has_nan": self.has_nan,
            "has_colors": self.has_colors,
            "has_normals": self.has_normals,
            "bbox_min": self.points[valid].min(axis=0).tolist() if n_valid > 0 else [],
            "bbox_max": self.points[valid].max(axis=0).tolist() if n_valid > 0 else [],
            "metadata": self.metadata,
        }

    def save_ply(self, path: str, binary: bool = True) -> None:
        """保存为 PLY 文件.

        Args:
            path: 输出文件路径.
            binary: True 用 binary (紧凑), False 用 ASCII (可读).
        """
        n = self.n_points
        has_c = self.has_colors
        with open(path, "wb") as f:
            # PLY header (ASCII).
            header_lines = [
                "ply",
                "format " + ("binary_little_endian" if binary else "ascii") + " 1.0",
                f"element vertex {n}",
                "property float x",
                "property float y",
                "property float z",
            ]
            if has_c:
                header_lines += [
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ]
            header_lines.append("end_header")
            header = "\n".join(header_lines) + "\n"
            f.write(header.encode("ascii"))
            # 数据.
            if binary:
                # 构建 dtype.
                dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
                if has_c:
                    dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
                dt = np.dtype(dtype_fields)
                arr = np.empty(n, dtype=dt)
                arr["x"] = self.points[:, 0]
                arr["y"] = self.points[:, 1]
                arr["z"] = self.points[:, 2]
                if has_c:
                    arr["red"] = self.colors[:, 0]
                    arr["green"] = self.colors[:, 1]
                    arr["blue"] = self.colors[:, 2]
                f.write(arr.tobytes())
            else:
                for i in range(n):
                    line = f"{self.points[i, 0]} {self.points[i, 1]} {self.points[i, 2]}"
                    if has_c:
                        line += f" {self.colors[i, 0]} {self.colors[i, 1]} {self.colors[i, 2]}"
                    line += "\n"
                    f.write(line.encode("ascii"))

    @classmethod
    def load_ply(cls, path: str) -> PointCloud:
        """从 PLY 文件加载."""
        with open(path, "rb") as f:
            data = f.read()
        # 解析 header.
        header_end = data.find(b"end_header\n") + len(b"end_header\n")
        header = data[:header_end].decode("ascii")
        body = data[header_end:]
        lines = header.split("\n")
        n_vertices = 0
        has_colors = False
        for line in lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if "red" in line:
                has_colors = True
        # 解析数据.
        is_binary = "binary_little_endian" in header
        if is_binary:
            dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_colors:
                dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
            dt = np.dtype(dtype_fields)
            arr = np.frombuffer(body, dtype=dt, count=n_vertices)
            points = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32)
            colors = (
                np.stack([arr["red"], arr["green"], arr["blue"]], axis=-1).astype(np.uint8)
                if has_colors
                else np.zeros((0, _CHANNELS_RGB), dtype=np.uint8)
            )
        else:
            # ASCII.
            text_lines = body.decode("ascii").strip().split("\n")
            points_list = []
            colors_list = []
            for line in text_lines[:n_vertices]:
                parts = line.split()
                points_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_colors:
                    colors_list.append([int(parts[3]), int(parts[4]), int(parts[5])])
            points = np.array(points_list, dtype=np.float32)
            colors = (
                np.array(colors_list, dtype=np.uint8)
                if has_colors
                else np.zeros((0, _CHANNELS_RGB), dtype=np.uint8)
            )
        return cls(points=points, colors=colors)
