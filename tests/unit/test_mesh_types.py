"""M017 单元测试: TriangleMesh 数据结构 + 导出.

覆盖:
- 构造 + 校验 (shape / 索引范围)
- 属性 (n_vertices / n_faces / has_*)
- 面法向量计算
- summary / has_nan
- PLY 保存/加载 (binary + ASCII)
- OBJ 导出
- glTF 导出
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from modules.mesh import TriangleMesh


def _make_simple_mesh(n_v: int = 4) -> TriangleMesh:
    """简单四面体 mesh."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return TriangleMesh(vertices=vertices, faces=faces)


class TestMeshConstruction:
    def test_basic_construction(self) -> None:
        mesh = _make_simple_mesh()
        assert mesh.n_vertices == 4
        assert mesh.n_faces == 4

    def test_with_colors_and_normals(self) -> None:
        v = np.zeros((3, 3), dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        colors = np.full((3, 3), 128, dtype=np.uint8)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = TriangleMesh(vertices=v, faces=f, vertex_colors=colors, vertex_normals=normals)
        assert mesh.has_vertex_colors
        assert mesh.has_vertex_normals

    def test_empty_mesh(self) -> None:
        mesh = TriangleMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
        )
        assert mesh.n_vertices == 0
        assert mesh.n_faces == 0

    def test_invalid_vertices_shape(self) -> None:
        with pytest.raises(ValueError, match="vertices"):
            TriangleMesh(
                vertices=np.zeros((4, 2), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
            )

    def test_invalid_faces_shape(self) -> None:
        with pytest.raises(ValueError, match="faces"):
            TriangleMesh(
                vertices=np.zeros((4, 3), dtype=np.float32),
                faces=np.zeros((2, 4), dtype=np.int32),
            )

    def test_face_index_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="face index"):
            TriangleMesh(
                vertices=np.zeros((3, 3), dtype=np.float32),
                faces=np.array([[0, 1, 5]], dtype=np.int32),
            )


class TestMeshProperties:
    def test_n_vertices_n_faces(self) -> None:
        mesh = _make_simple_mesh()
        assert mesh.n_vertices == 4
        assert mesh.n_faces == 4

    def test_has_nan_false(self) -> None:
        mesh = _make_simple_mesh()
        assert mesh.has_nan() is False

    def test_has_nan_true(self) -> None:
        vertices = np.array([[0, 0, 0], [1, np.nan, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        assert mesh.has_nan() is True

    def test_summary(self) -> None:
        mesh = _make_simple_mesh()
        s = mesh.summary()
        assert s["n_vertices"] == 4
        assert s["n_faces"] == 4
        assert s["has_nan"] is False
        assert s["hole_ratio"] == 0.0  # 默认


class TestFaceNormals:
    def test_compute_face_normals(self) -> None:
        # XY 平面的三角形, 法向量应朝 Z.
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        normals = mesh.compute_face_normals()
        assert normals.shape == (1, 3)
        # 叉积 (v1-v0) x (v2-v0) = (1,0,0) x (0,1,0) = (0,0,1).
        assert normals[0, 2] > 0.9

    def test_empty_face_normals(self) -> None:
        mesh = TriangleMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
        )
        normals = mesh.compute_face_normals()
        assert normals.shape == (0, 3)


class TestPLYExport:
    def test_save_load_binary(self, tmp_path) -> None:
        mesh = _make_simple_mesh()
        path = str(tmp_path / "mesh.ply")
        mesh.save_ply(path, binary=True)
        assert os.path.exists(path)
        loaded = TriangleMesh.load_ply(path)
        assert loaded.n_vertices == mesh.n_vertices
        assert loaded.n_faces == mesh.n_faces
        assert np.allclose(loaded.vertices, mesh.vertices)

    def test_save_load_ascii(self, tmp_path) -> None:
        mesh = _make_simple_mesh()
        path = str(tmp_path / "mesh_ascii.ply")
        mesh.save_ply(path, binary=False)
        loaded = TriangleMesh.load_ply(path)
        assert loaded.n_vertices == 4
        assert loaded.n_faces == 4
        assert np.allclose(loaded.vertices, mesh.vertices, atol=1e-4)

    def test_ply_with_colors(self, tmp_path) -> None:
        v = np.zeros((3, 3), dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        mesh = TriangleMesh(vertices=v, faces=f, vertex_colors=colors)
        path = str(tmp_path / "colored.ply")
        mesh.save_ply(path)
        loaded = TriangleMesh.load_ply(path)
        assert loaded.has_vertex_colors
        assert np.array_equal(loaded.vertex_colors, colors)


class TestOBJExport:
    def test_save_obj(self, tmp_path) -> None:
        mesh = _make_simple_mesh()
        path = str(tmp_path / "mesh.obj")
        mesh.save_obj(path)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # 应有 4 个 v 行 + 4 个 f 行.
        assert content.count("v ") == 4
        assert content.count("f ") == 4

    def test_obj_with_normals(self, tmp_path) -> None:
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = TriangleMesh(vertices=v, faces=f, vertex_normals=normals)
        path = str(tmp_path / "mesh_norm.obj")
        mesh.save_obj(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "vn " in content
        assert "//" in content  # f v//vn 格式


class TestGLTFExport:
    def test_save_gltf(self, tmp_path) -> None:
        mesh = _make_simple_mesh()
        path = str(tmp_path / "mesh.gltf")
        mesh.save_gltf(path)
        assert os.path.exists(path)
        # bin 文件也应有.
        bin_path = str(tmp_path / "mesh.bin")
        assert os.path.exists(bin_path)
        # 验证 JSON.
        with open(path, encoding="utf-8") as f:
            gltf = json.load(f)
        assert gltf["asset"]["version"] == "2.0"
        assert gltf["meshes"][0]["primitives"][0]["mode"] == 4  # TRIANGLES
        assert gltf["accessors"][0]["count"] == 4  # 4 vertices
        assert gltf["accessors"][1]["count"] == 4 * 3  # 4 faces * 3 indices

    def test_gltf_min_max(self, tmp_path) -> None:
        v = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        f = np.array([[0, 1, 0]], dtype=np.int32)
        mesh = TriangleMesh(vertices=v, faces=f)
        path = str(tmp_path / "mesh.gltf")
        mesh.save_gltf(path)
        with open(path, encoding="utf-8") as f:
            gltf = json.load(f)
        pos_accessor = gltf["accessors"][0]
        assert pos_accessor["min"] == [0.0, 0.0, 0.0]
        assert pos_accessor["max"] == [1.0, 2.0, 3.0]
