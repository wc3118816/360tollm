"""M017 Mesh 重建: TriangleMesh 数据结构.

承载:
- vertices: float32 (V, 3) 三角网格顶点 (世界坐标系).
- faces: int32 (F, 3) 三角形顶点索引 (每行 3 个顶点索引).
- vertex_normals: float32 (V, 3) 顶点法向量 (可选).
- vertex_colors: uint8 (V, 3) 顶点 RGB 颜色 (可选).
- face_normals: float32 (F, 3) 面法向量 (可选, 由叉积计算).
- metadata: source_frame_id / projection / n_faces / hole_ratio 等.

导出:
- save_ply() / load_ply():  PLY (vertices + faces, binary + ASCII).
- save_obj():                OBJ (v / vn / f).
- save_gltf():               glTF 2.0 (JSON + embedded binary buffer).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_NDIM_2D = 2
_NDIM_3D = 3
_CHANNELS_RGB = 3
_FACE_SIZE = 3
# 归一化最小阈值 (避免除零).
_NORM_EPS = 1e-12


@dataclass(slots=True)
class TriangleMesh:
    """三角网格.

    Attributes:
        vertices: float32 (V, 3) 顶点坐标.
        faces: int32 (F, 3) 三角形顶点索引.
        vertex_normals: float32 (V, 3) 顶点法向量, 可选.
        vertex_colors: uint8 (V, 3) 顶点颜色, 可选.
        metadata: 扩展字段 (source / projection / n_faces / hole_ratio 等).
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray = field(
        default_factory=lambda: np.zeros((0, _NDIM_3D), dtype=np.float32)
    )
    vertex_colors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, _CHANNELS_RGB), dtype=np.uint8)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        if self.vertices.ndim != _NDIM_2D or self.vertices.shape[1] != _NDIM_3D:
            raise ValueError(f"vertices must be (V,3), got {self.vertices.shape}")
        self.faces = np.asarray(self.faces, dtype=np.int32)
        if self.faces.ndim != _NDIM_2D or self.faces.shape[1] != _FACE_SIZE:
            raise ValueError(f"faces must be (F,3), got {self.faces.shape}")
        # 索引范围检查.
        n_v = self.vertices.shape[0]
        if n_v > 0 and self.faces.shape[0] > 0:
            if self.faces.max() >= n_v:
                raise ValueError(f"face index {int(self.faces.max())} >= n_vertices {n_v}")
            if self.faces.min() < 0:
                raise ValueError(f"face index {int(self.faces.min())} < 0")
        # 法向量.
        self.vertex_normals = np.asarray(self.vertex_normals, dtype=np.float32)
        if self.vertex_normals.size == 0:
            self.vertex_normals = self.vertex_normals.reshape(0, _NDIM_3D)
        # 颜色.
        self.vertex_colors = np.asarray(self.vertex_colors, dtype=np.uint8)
        if self.vertex_colors.size == 0:
            self.vertex_colors = self.vertex_colors.reshape(0, _CHANNELS_RGB)

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def has_vertex_normals(self) -> bool:
        return self.vertex_normals.shape[0] == self.n_vertices and self.n_vertices > 0

    @property
    def has_vertex_colors(self) -> bool:
        return self.vertex_colors.shape[0] == self.n_vertices and self.n_vertices > 0

    def compute_face_normals(self) -> np.ndarray:
        """计算面法向量 (叉积). Returns (F, 3) float32."""
        if self.n_faces == 0 or self.n_vertices == 0:
            return np.zeros((0, _NDIM_3D), dtype=np.float32)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > _NORM_EPS, norms, 1.0)
        return (normals / norms).astype(np.float32)

    def summary(self) -> dict[str, Any]:
        """统计摘要."""
        has_nan = bool(np.any(~np.isfinite(self.vertices))) if self.n_vertices > 0 else False
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "has_vertex_normals": self.has_vertex_normals,
            "has_vertex_colors": self.has_vertex_colors,
            "has_nan": has_nan,
            "hole_ratio": self.metadata.get("hole_ratio", 0.0),
            "metadata": self.metadata,
        }

    def has_nan(self) -> bool:
        """检查顶点是否含 NaN/Inf."""
        return bool(np.any(~np.isfinite(self.vertices))) if self.n_vertices > 0 else False

    # ---- PLY ----

    def save_ply(self, path: str, binary: bool = True) -> None:
        """保存 PLY (vertices + faces)."""
        n_v = self.n_vertices
        n_f = self.n_faces
        has_color = self.has_vertex_colors
        has_normal = self.has_vertex_normals

        # header.
        header_lines = ["ply", "format " + ("binary_little_endian 1.0" if binary else "ascii 1.0")]
        elem_v = [
            f"element vertex {n_v}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_normal:
            elem_v += ["property float nx", "property float ny", "property float nz"]
        if has_color:
            elem_v += ["property uchar red", "property uchar green", "property uchar blue"]
        header_lines += elem_v
        header_lines += [
            f"element face {n_f}",
            "property list uchar int vertex_indices",
            "end_header",
        ]

        if not binary:
            # ASCII.
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(header_lines) + "\n")
                for i in range(n_v):
                    parts = [
                        f"{self.vertices[i, 0]:.6f}",
                        f"{self.vertices[i, 1]:.6f}",
                        f"{self.vertices[i, 2]:.6f}",
                    ]
                    if has_normal:
                        parts += [
                            f"{self.vertex_normals[i, 0]:.6f}",
                            f"{self.vertex_normals[i, 1]:.6f}",
                            f"{self.vertex_normals[i, 2]:.6f}",
                        ]
                    if has_color:
                        parts += [
                            str(int(self.vertex_colors[i, 0])),
                            str(int(self.vertex_colors[i, 1])),
                            str(int(self.vertex_colors[i, 2])),
                        ]
                    f.write(" ".join(parts) + "\n")
                for face in self.faces:
                    f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
            return

        # Binary.
        with open(path, "wb") as f:
            f.write(("\n".join(header_lines) + "\n").encode("ascii"))
            # vertex data.
            v_dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_normal:
                v_dtype_fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
            if has_color:
                v_dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
            v_arr = np.zeros(n_v, dtype=v_dtype_fields)
            v_arr["x"] = self.vertices[:, 0]
            v_arr["y"] = self.vertices[:, 1]
            v_arr["z"] = self.vertices[:, 2]
            if has_normal:
                v_arr["nx"] = self.vertex_normals[:, 0]
                v_arr["ny"] = self.vertex_normals[:, 1]
                v_arr["nz"] = self.vertex_normals[:, 2]
            if has_color:
                v_arr["red"] = self.vertex_colors[:, 0]
                v_arr["green"] = self.vertex_colors[:, 1]
                v_arr["blue"] = self.vertex_colors[:, 2]
            f.write(v_arr.tobytes())
            # face data: uchar count (3) + 3 int32.
            for face in self.faces:
                f.write(struct.pack("<B", 3))
                f.write(struct.pack("<iii", int(face[0]), int(face[1]), int(face[2])))

    @classmethod
    def load_ply(cls, path: str) -> TriangleMesh:  # noqa: PLR0912, PLR0915
        """加载 PLY (vertices + faces)."""
        with open(path, "rb") as f:
            data = f.read()
        # 解析 header.
        header_end = data.find(b"end_header")
        if header_end == -1:
            raise ValueError("PLY: no end_header found")
        header_text = data[:header_end].decode("ascii")
        body = data[header_end + len(b"end_header") + 1 :]

        n_v = 0
        n_f = 0
        has_color = False
        has_normal = False
        is_binary = "binary" in header_text
        for line in header_text.splitlines():
            if line.startswith("element vertex"):
                n_v = int(line.split()[-1])
            elif line.startswith("element face"):
                n_f = int(line.split()[-1])
            elif line.startswith("property uchar red"):
                has_color = True
            elif line.startswith("property float nx"):
                has_normal = True

        if not is_binary:
            # ASCII.
            lines = body.decode("ascii").strip().splitlines()
            vertices = []
            vertex_normals = []
            vertex_colors = []
            for i in range(n_v):
                parts = lines[i].split()
                idx = 0
                v = [float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])]
                idx += 3
                vertices.append(v)
                if has_normal:
                    vertex_normals.append(
                        [float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])]
                    )
                    idx += 3
                if has_color:
                    vertex_colors.append(
                        [int(parts[idx]), int(parts[idx + 1]), int(parts[idx + 2])]
                    )
                    idx += 3
            faces = []
            for j in range(n_f):
                parts = lines[n_v + j].split()
                # parts[0] = count (3), 不再读取.
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
            return cls(
                vertices=np.array(vertices, dtype=np.float32),
                faces=np.array(faces, dtype=np.int32),
                vertex_normals=np.array(vertex_normals, dtype=np.float32)
                if vertex_normals
                else np.zeros((0, 3), dtype=np.float32),
                vertex_colors=np.array(vertex_colors, dtype=np.uint8)
                if vertex_colors
                else np.zeros((0, 3), dtype=np.uint8),
            )

        # Binary.
        offset = 0
        v_dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if has_normal:
            v_dtype_fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        if has_color:
            v_dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        v_arr = np.frombuffer(body, dtype=v_dtype_fields, count=n_v, offset=offset)
        offset += n_v * v_arr.itemsize
        vertices = np.stack([v_arr["x"], v_arr["y"], v_arr["z"]], axis=1).astype(np.float32)
        vertex_normals = np.zeros((0, 3), dtype=np.float32)
        if has_normal:
            vertex_normals = np.stack([v_arr["nx"], v_arr["ny"], v_arr["nz"]], axis=1).astype(
                np.float32
            )
        vertex_colors = np.zeros((0, 3), dtype=np.uint8)
        if has_color:
            vertex_colors = np.stack([v_arr["red"], v_arr["green"], v_arr["blue"]], axis=1).astype(
                np.uint8
            )
        # faces: uchar count + 3 int32 per face.
        face_size = 1 + 3 * 4
        faces = np.zeros((n_f, 3), dtype=np.int32)
        for i in range(n_f):
            base = offset + i * face_size
            # body[base] = count (3), 不再读取.
            vals = struct.unpack_from("<iii", body, base + 1)
            faces[i] = vals
        return cls(
            vertices=vertices,
            faces=faces,
            vertex_normals=vertex_normals,
            vertex_colors=vertex_colors,
        )

    # ---- OBJ ----

    def save_obj(self, path: str) -> None:
        """保存 OBJ (v / vn / f)."""
        has_normal = self.has_vertex_normals
        with open(path, "w", encoding="utf-8") as f:
            f.write("# M017 TriangleMesh\n")
            # vertices (1-indexed in OBJ).
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            # normals.
            if has_normal:
                for n in self.vertex_normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            # faces (1-indexed).
            for face in self.faces:
                i0, i1, i2 = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
                if has_normal:
                    f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
                else:
                    f.write(f"f {i0} {i1} {i2}\n")

    # ---- glTF 2.0 ----

    def save_gltf(self, path: str) -> None:
        """保存 glTF 2.0 (JSON + embedded binary buffer).

        结构:
        - buffer: vertices (float32) + indices (uint32), little-endian.
        - bufferView 0: vertices (ARRAY_BUFFER).
        - bufferView 1: indices (ELEMENT_ARRAY_BUFFER).
        - accessors: position (VEC3) + indices (SCALAR).
        - mesh.primitives: POSITION + indices.
        """
        n_v = self.n_vertices
        n_f = self.n_faces

        # 二进制数据: vertices + indices.
        vert_bytes = self.vertices.astype("<f4").tobytes()
        idx_bytes = self.faces.astype("<u4").tobytes()
        bin_data = vert_bytes + idx_bytes

        # glTF bin 文件.
        bin_path = path.rsplit(".", 1)[0] + ".bin"
        with open(bin_path, "wb") as bf:
            bf.write(bin_data)

        # glTF JSON.
        gltf = {
            "asset": {"version": "2.0", "generator": "360tollm M017"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0},
                            "indices": 1,
                            "mode": 4,  # TRIANGLES
                        }
                    ]
                }
            ],
            "buffers": [
                {
                    "uri": bin_path.split("\\")[-1].split("/")[-1],
                    "byteLength": len(bin_data),
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(vert_bytes),
                    "target": 34962,  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": len(vert_bytes),
                    "byteLength": len(idx_bytes),
                    "target": 34963,  # ELEMENT_ARRAY_BUFFER
                },
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": n_v,
                    "type": "VEC3",
                    "min": [float(self.vertices[:, i].min()) for i in range(3)],
                    "max": [float(self.vertices[:, i].max()) for i in range(3)],
                },
                {
                    "bufferView": 1,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": n_f * 3,
                    "type": "SCALAR",
                },
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gltf, f, indent=2)
